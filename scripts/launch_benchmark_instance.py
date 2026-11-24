#!/usr/bin/env python3
"""
Launch an EC2 instance configured for dataloader benchmarking.

Targets a CPU-focused Ubuntu AMI,
installs bootstrap tooling for Docker + uv, and ships helper scripts for dataset
and log management.
"""

from __future__ import annotations

import gzip
import logging
import pathlib
import shlex
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import boto3
import click
from botocore.exceptions import ClientError

LOGGER = logging.getLogger("launch_benchmark")
LOGGER.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.INFO)
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
)
LOGGER.addHandler(_handler)

REGION_NAME = "us-east-1"
DEFAULT_INSTANCE_TYPE = "r8id.48xlarge"

DEFAULT_SUBNET_TAG = None
DEFAULT_KEY_PAIR_NAME = None
DEFAULT_IAM_INSTANCE_PROFILE = None

UBUNTU_OWNER_ID = "099720109477"
UBUNTU_AMI_PATTERNS = {
    "24.04": (
        "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*",
        "ubuntu/images/hvm-ssd/ubuntu-noble-24.04-amd64-server-*",
    ),
    "22.04": (
        "ubuntu/images/hvm-ssd-gp3/ubuntu-jammy-22.04-amd64-server-*",
        "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*",
    ),
}
DEFAULT_UBUNTU_VERSION = "24.04"

DEFAULT_BOOTSTRAP_SCRIPT = pathlib.Path("scripts/bootstrap_benchmark.sh")
DEFAULT_OPEN_FILE_LIMIT = 65536

SMT_DISABLE_RUNTIME_SNIPPET = """
# --- Start SMT (Hyperthreading) Runtime Disable ---
echo "Disabling SMT at the OS level (for bare-metal instances)..."

if [ -f /sys/devices/system/cpu/smt/control ]; then
    echo off > /sys/devices/system/cpu/smt/control
    echo "SMT disabled at runtime via sysfs."
else
    echo "WARNING: /sys/devices/system/cpu/smt/control not found."
fi
# --- End SMT (Hyperthreading) Runtime Disable ---
"""

SMT_DISABLE_GRUB_SNIPPET = """
# --- Start SMT (Hyperthreading) GRUB Persist ---
sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/&nosmt /' /etc/default/grub

if [ -f "/etc/default/grub.d/50-cloudimg-settings.cfg" ]; then
    echo 'GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT} nosmt"' >> /etc/default/grub.d/50-cloudimg-settings.cfg
fi

update-grub
echo "SMT GRUB persistence configured. Reboot required."
# --- End SMT (Hyperthreading) GRUB Persist ---
"""

CSTATE_SETUP_SNIPPET = """
# --- Start C-State Restriction Setup ---
echo "Configuring GRUB for C-State restriction (intel_idle.max_cstate=1)..."

# 1. Update standard /etc/default/grub
# Prepend flags to the existing quoted string to ensure we don't overwrite other defaults
sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/&intel_idle.max_cstate=1 processor.max_cstate=1 /' /etc/default/grub

# 2. Update Cloud-init override (common on AWS Ubuntu AMIs)
if [ -f "/etc/default/grub.d/50-cloudimg-settings.cfg" ]; then
    echo 'GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT} intel_idle.max_cstate=1 processor.max_cstate=1"' >> /etc/default/grub.d/50-cloudimg-settings.cfg
fi

update-grub
echo "GRUB updated. A reboot is required for C-State restrictions to take effect."
# --- End C-State Restriction Setup ---
"""

PERF_STABILIZE_SNIPPET = r"""
# --- Start Performance Stabilization Setup ---
echo "Installing CPU performance stabilization service..."

cat >/usr/local/sbin/stabilize-benchmarks.sh <<'STABSCRIPT'
#!/usr/bin/env bash
set -uo pipefail

echo "=== Benchmark performance stabilization ==="

# 1. CPU governor -> performance (fixed frequency, no scaling)
# Offline CPUs may fail; that's fine.
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -f "$gov" ] && echo performance > "$gov" 2>/dev/null || true
done
echo "CPU governor set to performance."

# 2. Disable turbo boost (Intel and AMD)
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
    echo "Intel turbo boost disabled."
elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo 0 > /sys/devices/system/cpu/cpufreq/boost
    echo "AMD boost disabled."
fi

# 3. Disable NMI watchdog (reduces interrupt jitter)
if [ -f /proc/sys/kernel/nmi_watchdog ]; then
    echo 0 > /proc/sys/kernel/nmi_watchdog
    echo "NMI watchdog disabled."
fi

# 4. Disable NUMA balancing (avoids page migration noise on multi-socket)
if [ -f /proc/sys/kernel/numa_balancing ]; then
    echo 0 > /proc/sys/kernel/numa_balancing
    echo "NUMA balancing disabled."
fi

# 5. THP: enabled with deferred defrag (huge pages without compaction stalls)
if [ -d /sys/kernel/mm/transparent_hugepage ]; then
    echo always > /sys/kernel/mm/transparent_hugepage/enabled
    echo defer > /sys/kernel/mm/transparent_hugepage/defrag
    echo "THP enabled, defrag set to defer."
fi

# 6. Disable ASLR (reduces memory layout variance between runs)
if [ -f /proc/sys/kernel/randomize_va_space ]; then
    echo 0 > /proc/sys/kernel/randomize_va_space
    echo "ASLR disabled."
fi

# 7. Disable scheduler autogroups (no TTY-based grouping)
if [ -f /proc/sys/kernel/sched_autogroup_enabled ]; then
    echo 0 > /proc/sys/kernel/sched_autogroup_enabled
    echo "Scheduler autogroups disabled."
fi

# 8. Disable IRQ balancing
if systemctl is-active --quiet irqbalance 2>/dev/null; then
    systemctl stop irqbalance
    systemctl disable irqbalance
    echo "irqbalance stopped and disabled."
fi

# 9. Prevent swap consideration
if [ -f /proc/sys/vm/swappiness ]; then
    echo 0 > /proc/sys/vm/swappiness
    echo "vm.swappiness set to 0."
fi

# 10. Disable timer migration (reduces cross-core jitter)
if [ -f /proc/sys/kernel/timer_migration ]; then
    echo 0 > /proc/sys/kernel/timer_migration
    echo "Timer migration disabled."
fi

# 11. I/O scheduler -> none for NVMe (no overhead)
for sched in /sys/block/nvme*/queue/scheduler; do
    [ -f "$sched" ] && echo none > "$sched" 2>/dev/null || true
done
echo "I/O scheduler set to none for NVMe devices."

# 12. Disable background apt/fwupd services that cause random I/O and CPU spikes.
# Keep snapd enabled: on Ubuntu 24.04, amazon-ssm-agent is snap-based.
for svc in \
    apt-daily.timer apt-daily-upgrade.timer \
    unattended-upgrades.service \
    fwupd-refresh.timer; do
    if systemctl is-enabled --quiet "$svc" 2>/dev/null; then
        systemctl stop "$svc" 2>/dev/null || true
        systemctl disable "$svc" 2>/dev/null || true
    fi
done
echo "Background services (apt-daily, fwupd) disabled; snapd kept enabled for SSM."

echo "=== Stabilization complete ==="
STABSCRIPT

chmod 0755 /usr/local/sbin/stabilize-benchmarks.sh

cat >/etc/systemd/system/stabilize-benchmarks.service <<'STABUNIT'
[Unit]
Description=Apply CPU and memory tuning for stable benchmarks
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
TimeoutStartSec=60
ExecStart=/usr/local/sbin/stabilize-benchmarks.sh

[Install]
WantedBy=multi-user.target
STABUNIT

systemctl daemon-reload
systemctl enable stabilize-benchmarks.service
systemctl start stabilize-benchmarks.service
echo "Performance stabilization service installed and started."
# --- End Performance Stabilization Setup ---
"""

OPEN_FILE_LIMIT_SNIPPET = """
# --- Start Open File Limit Setup ---
echo "Configuring system-wide open file limit to {limit}"

cat >/etc/security/limits.d/99-nofile.conf <<'EOF'
* soft nofile {limit}
* hard nofile {limit}
root soft nofile {limit}
root hard nofile {limit}
EOF

mkdir -p /etc/systemd/system.conf.d /etc/systemd/user.conf.d

cat >/etc/systemd/system.conf.d/99-nofile.conf <<'EOF'
[Manager]
DefaultLimitNOFILE={limit}
EOF

cat >/etc/systemd/user.conf.d/99-nofile.conf <<'EOF'
[Manager]
DefaultLimitNOFILE={limit}
EOF

systemctl daemon-reexec || true
ulimit -n {limit} || true
echo "Open file limits configured."
# --- End Open File Limit Setup ---
"""


class LaunchError(RuntimeError):
    """Raised when provisioning fails."""


@dataclass(frozen=True)
class LaunchConfig:
    instance_name: str
    instance_type: str
    key_pair_name: Optional[str]
    security_group_name: str
    subnet_tag: Optional[str]
    ami_pattern: Optional[str]
    ami_owner: str
    ami_id: Optional[str]
    iam_instance_profile: Optional[str]
    root_volume_size: int
    shutdown_behavior: str
    user_data: Optional[str]
    tags: dict[str, str]
    restrict_cstates: bool
    open_file_limit: int
    disable_smt: bool
    core_count: Optional[int]
    stabilize_performance: bool
    ubuntu_version: str


def _ec2_client():
    return boto3.client("ec2", region_name=REGION_NAME)


def _ec2_resource():
    return boto3.resource("ec2", region_name=REGION_NAME)


def _ssm_client():
    return boto3.client("ssm", region_name=REGION_NAME)


def resolve_subnet_id(subnet_tag: str) -> str:
    LOGGER.info("Resolving subnet ID for tag name '%s'", subnet_tag)
    response = _ec2_client().describe_subnets(
        Filters=[{"Name": "tag:Name", "Values": [subnet_tag]}]
    )
    subnets = response.get("Subnets", [])
    if not subnets:
        raise LaunchError(f"No subnet found with tag: {subnet_tag}")
    if len(subnets) > 1:
        LOGGER.warning(
            "Multiple subnets matched tag '%s'; using first result %s",
            subnet_tag,
            subnets[0]["SubnetId"],
        )
    return subnets[0]["SubnetId"]


def resolve_security_group_id(group_name: str) -> str:
    LOGGER.info("Resolving security group ID for name '%s'", group_name)
    response = _ec2_client().describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [group_name]}]
    )
    groups = response.get("SecurityGroups", [])
    if not groups:
        raise LaunchError(f"No security group found named: {group_name}")
    if len(groups) > 1:
        LOGGER.warning(
            "Multiple security groups matched '%s'; using first result %s",
            group_name,
            groups[0]["GroupId"],
        )
    return groups[0]["GroupId"]


SSM_UBUNTU_PARAMS = {
    "24.04": [
        "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id",
        "/aws/service/canonical/ubuntu/server/noble/stable/current/amd64/hvm/ebs-gp3/ami-id",
        "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp2/ami-id",
        "/aws/service/canonical/ubuntu/server/noble/stable/current/amd64/hvm/ebs-gp2/ami-id",
    ],
    "22.04": [
        "/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp3/ami-id",
        "/aws/service/canonical/ubuntu/server/jammy/stable/current/amd64/hvm/ebs-gp3/ami-id",
        "/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id",
        "/aws/service/canonical/ubuntu/server/jammy/stable/current/amd64/hvm/ebs-gp2/ami-id",
    ],
}


def resolve_ami_via_ssm(ubuntu_version: str) -> Optional[str]:
    """Try AWS SSM public parameters for Ubuntu LTS AMI."""
    param_candidates = SSM_UBUNTU_PARAMS.get(ubuntu_version, SSM_UBUNTU_PARAMS["24.04"])
    ssm = _ssm_client()
    for name in param_candidates:
        try:
            resp = ssm.get_parameter(Name=name)
            value = resp.get("Parameter", {}).get("Value")
            if value:
                LOGGER.info("Resolved AMI via SSM parameter %s: %s", name, value)
                return value
        except ClientError:
            # Try next candidate
            continue
    return None


def resolve_latest_ami(ami_patterns: Iterable[str], owner: str) -> str:
    client = _ec2_client()
    for pattern in ami_patterns:
        LOGGER.info("Resolving latest AMI for owner=%s pattern=%s", owner, pattern)
        response = client.describe_images(
            Owners=[owner],
            Filters=[
                {"Name": "name", "Values": [pattern]},
                {"Name": "architecture", "Values": ["x86_64"]},
                {"Name": "virtualization-type", "Values": ["hvm"]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "root-device-type", "Values": ["ebs"]},
            ],
        )
        images = response.get("Images", [])
        if not images:
            continue
        images.sort(key=lambda img: img["CreationDate"], reverse=True)
        latest = images[0]
        LOGGER.info(
            "Selected AMI %s created at %s",
            latest["ImageId"],
            latest["CreationDate"],
        )
        return latest["ImageId"]
    raise LaunchError(f"No AMIs found for owner={owner} with provided patterns")


def load_user_data(path: Optional[pathlib.Path]) -> Optional[str]:
    if path is None:
        return None
    if not path.exists():
        raise LaunchError(f"User-data script not found at {path}")
    LOGGER.info("Loading user-data from %s", path)
    return path.read_text()


def encode_user_data(user_data: Optional[str]) -> Optional[bytes]:
    if not user_data:
        return None
    return gzip.compress(user_data.encode("utf-8"))


def _is_metal_instance(instance_type: str) -> bool:
    """Return True if the instance type is a bare-metal instance."""
    # Metal instances have size components like "metal", "metal-24xl", "metal-48xl", etc.
    parts = instance_type.split(".")
    return len(parts) == 2 and parts[1].startswith("metal")


def resolve_default_core_count(instance_type: str) -> int:
    """Query AWS for the default number of physical cores for an instance type."""
    LOGGER.info("Resolving default core count for %s...", instance_type)
    client = _ec2_client()
    try:
        response = client.describe_instance_types(InstanceTypes=[instance_type])
        core_count = response["InstanceTypes"][0]["VCpuInfo"]["DefaultCores"]
        LOGGER.info(
            "Found %d default physical cores for %s.", core_count, instance_type
        )
        return core_count
    except ClientError as exc:
        raise LaunchError(
            f"Failed to lookup instance type {instance_type}: {exc}"
        ) from exc
    except (KeyError, IndexError):
        raise LaunchError(
            f"Could not parse core count for {instance_type} from AWS response."
        )


def create_instance(config: LaunchConfig) -> tuple[str, str]:
    subnet_id = resolve_subnet_id(config.subnet_tag) if config.subnet_tag else None
    security_group_id = resolve_security_group_id(config.security_group_name)
    if config.ami_id:
        ami_id = config.ami_id
        LOGGER.info("Using explicit AMI ID: %s", ami_id)
    elif config.ami_pattern:
        ami_id = resolve_latest_ami([config.ami_pattern], config.ami_owner)
    else:
        ami_patterns = UBUNTU_AMI_PATTERNS.get(
            config.ubuntu_version, UBUNTU_AMI_PATTERNS[DEFAULT_UBUNTU_VERSION]
        )
        ami_id = resolve_ami_via_ssm(config.ubuntu_version) or resolve_latest_ami(
            ami_patterns, config.ami_owner
        )

    ec2 = _ec2_resource()
    LOGGER.info(
        "Launching instance %s (%s)",
        config.instance_name,
        config.instance_type,
    )

    block_device_mappings = [
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "DeleteOnTermination": True,
                "VolumeSize": config.root_volume_size,
                "VolumeType": "gp3",
            },
        }
    ]

    launch_kwargs = {
        "ImageId": ami_id,
        "MinCount": 1,
        "MaxCount": 1,
        "InstanceType": config.instance_type,
        "SecurityGroupIds": [security_group_id],
    }

    if config.key_pair_name:
        launch_kwargs["KeyName"] = config.key_pair_name
    if config.iam_instance_profile:
        launch_kwargs["IamInstanceProfile"] = {"Name": config.iam_instance_profile}

    if subnet_id:
        launch_kwargs["SubnetId"] = subnet_id
    else:
        LOGGER.info(
            "No subnet specified; AWS will select an AZ with available capacity."
        )

    launch_kwargs.update(
        {
            "BlockDeviceMappings": block_device_mappings,
            "InstanceInitiatedShutdownBehavior": config.shutdown_behavior,
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": key, "Value": value}
                        for key, value in config.tags.items()
                    ],
                }
            ],
        }
    )

    if config.disable_smt and not _is_metal_instance(config.instance_type):
        if not config.core_count:
            raise LaunchError(
                "AWS requires you to specify --core-count when disabling SMT."
            )
        LOGGER.info(
            "Disabling SMT: Requesting %s physical cores with 1 thread per core.",
            config.core_count,
        )
        launch_kwargs["CpuOptions"] = {
            "CoreCount": config.core_count,
            "ThreadsPerCore": 1,
        }

    if config.user_data:
        launch_kwargs["UserData"] = encode_user_data(config.user_data)

    try:
        instances = ec2.create_instances(**launch_kwargs)
    except ClientError as exc:
        raise LaunchError(f"Failed to launch instance: {exc}") from exc

    if len(instances) != 1:
        raise LaunchError(f"Expected a single instance, got {len(instances)}")

    instance = instances[0]
    LOGGER.info("Waiting for instance %s to enter running state", instance.id)
    instance.wait_until_running()
    instance.reload()
    return instance.id, instance.private_ip_address or ""


def format_ssh_config(instance_id: str, key_pair_name: str | None) -> str:
    identity_line = f"IdentityFile ~/.ssh/{key_pair_name}.pem" if key_pair_name else ""
    template = f"""
Host benchmark-{instance_id}
    HostName {instance_id}
    User ubuntu
    {identity_line}
    ForwardAgent yes
    ProxyCommand aws --region {REGION_NAME} ssm start-session --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p
"""
    return template.strip()


def output_launch_details(
    instance_name: str,
    instance_id: str,
    private_ip: str,
    key_pair_name: Optional[str],
    iam_instance_profile: Optional[str],
) -> None:
    instance_url = (
        f"https://{REGION_NAME}.console.aws.amazon.com/ec2/home"
        f"?region={REGION_NAME}#InstanceDetails:instanceId={instance_id}"
    )

    LOGGER.info("Launched EC2 Instance %s", instance_name)
    LOGGER.info("Instance ID: %s", instance_id)
    if private_ip:
        LOGGER.info("Private IP: %s", private_ip)
    LOGGER.info("Key Pair: %s", key_pair_name)
    LOGGER.info("Instance URL: %s", instance_url)

    if key_pair_name and iam_instance_profile:
        LOGGER.info(
            "SSH configuration snippet:\n%s",
            format_ssh_config(instance_id, key_pair_name),
        )
    elif key_pair_name:
        LOGGER.info(
            "SSH over your VPC connection: ssh -A -i ~/.ssh/%s.pem ubuntu@%s",
            key_pair_name,
            private_ip,
        )
    else:
        LOGGER.info(
            "Connect with Session Manager: aws --region %s ssm start-session --target %s",
            REGION_NAME,
            instance_id,
        )


def collect_tags(extra_tags: Iterable[str], instance_name: str) -> dict[str, str]:
    tags = {"Name": instance_name, "Role": "dataloader-benchmark"}
    for tag in extra_tags:
        if "=" not in tag:
            raise LaunchError(f"Tag '{tag}' must be in key=value format")
        key, value = tag.split("=", 1)
        tags[key] = value
    return tags


def _inject_snippet(user_data: Optional[str], snippet: str) -> str:
    """Insert a shell snippet after any shebang while preserving the rest of the script."""
    snippet = snippet.strip("\n")
    if not user_data:
        return f"#!/bin/bash\n{snippet}\n"

    if user_data.startswith("#!"):
        lines = user_data.splitlines()
        return "\n".join([lines[0], snippet] + lines[1:])

    return snippet + "\n" + user_data


def inject_cstate_setup(user_data: Optional[str]) -> str:
    """Prepend GRUB configuration logic to the user-data script."""
    return _inject_snippet(user_data, CSTATE_SETUP_SNIPPET)


def inject_smt_disable(user_data: Optional[str], include_grub: bool = True) -> str:
    """Prepend OS-level SMT disable logic to user-data (for bare-metal instances)."""
    user_data = _inject_snippet(user_data, SMT_DISABLE_RUNTIME_SNIPPET)
    if include_grub:
        user_data = _inject_snippet(user_data, SMT_DISABLE_GRUB_SNIPPET)
    return user_data


def inject_perf_stabilize(user_data: Optional[str]) -> str:
    """Install a systemd service that pins CPU governor, disables turbo, etc."""
    return _inject_snippet(user_data, PERF_STABILIZE_SNIPPET)


def inject_open_file_limit(user_data: Optional[str], open_file_limit: int) -> str:
    """Ensure open file limits are raised early in user-data execution."""
    snippet = OPEN_FILE_LIMIT_SNIPPET.format(limit=open_file_limit)
    return _inject_snippet(user_data, snippet)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("instance_name", type=str)
@click.option("--region", default="us-east-1", show_default=True, help="AWS region.")
@click.option(
    "--repo-url",
    default="https://github.com/datologyai/dataloader-arena-public.git",
    show_default=True,
    help="Repository cloned by bootstrap and its login helper.",
)
@click.option(
    "--instance-type",
    default=DEFAULT_INSTANCE_TYPE,
    show_default=True,
    help="EC2 instance type to launch.",
)
@click.option(
    "--key-pair-name",
    default=DEFAULT_KEY_PAIR_NAME,
    show_default=True,
    help="Existing EC2 key pair name for SSH access.",
)
@click.option(
    "--security-group",
    required=True,
    show_default=True,
    help="Security group name to attach.",
)
@click.option(
    "--subnet-tag",
    default=DEFAULT_SUBNET_TAG,
    show_default=True,
    help="Subnet Name tag value used to resolve SubnetId.",
)
@click.option(
    "--ami-pattern",
    default=None,
    show_default=True,
    help="Name pattern used to resolve the AMI.",
)
@click.option(
    "--ami-owner",
    default=UBUNTU_OWNER_ID,
    show_default=True,
    help="Owner account ID for the AMI.",
)
@click.option(
    "--ami-id",
    default=None,
    help="Override AMI ID directly (skips resolution).",
)
@click.option(
    "--iam-instance-profile",
    default=DEFAULT_IAM_INSTANCE_PROFILE,
    show_default=True,
    help="IAM instance profile name to attach.",
)
@click.option(
    "--root-volume-size",
    default=200,
    show_default=True,
    type=int,
    help="Root EBS volume size in GiB.",
)
@click.option(
    "--keep-on-stop",
    is_flag=True,
    help="Configure the instance to stop instead of terminate on shutdown.",
)
@click.option(
    "--user-data-file",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=DEFAULT_BOOTSTRAP_SCRIPT,
    show_default=True,
    help="Path to a shell script used as EC2 user-data.",
)
@click.option(
    "--tag",
    "extra_tags",
    multiple=True,
    help="Additional tag in key=value format (can repeat).",
)
@click.option(
    "--restrict-cstates/--no-restrict-cstates",
    default=True,
    show_default=True,
    help="Configure GRUB to restrict CPU C-states (requires reboot).",
)
@click.option(
    "--open-file-limit",
    default=DEFAULT_OPEN_FILE_LIMIT,
    show_default=True,
    type=int,
    help="System-wide nofile limit to set via user-data (use 0 to skip).",
)
@click.option(
    "--disable-smt",
    is_flag=True,
    help="Disable SMT (Hyperthreading) by setting ThreadsPerCore=1. Requires --core-count.",
)
@click.option(
    "--core-count",
    type=int,
    default=None,
    help="Number of physical cores to allocate. Required if --disable-smt is used.",
)
@click.option(
    "--stabilize-performance/--no-stabilize-performance",
    default=True,
    show_default=True,
    help="Pin CPU governor to performance, disable turbo boost, reduce jitter.",
)
@click.option(
    "--disable-grub",
    is_flag=True,
    help="Skip all GRUB/kernel-param modifications (c-states, nosmt). Use runtime-only tuning.",
)
@click.option(
    "--bare-bones",
    is_flag=True,
    help="Skip all custom systemd services (RAID, stabilize, repo clone). Vanilla bootstrap only.",
)
@click.option(
    "--ubuntu-version",
    type=click.Choice(["22.04", "24.04"]),
    default=DEFAULT_UBUNTU_VERSION,
    show_default=True,
    help="Ubuntu LTS version to use.",
)
def main(
    instance_name: str,
    instance_type: str,
    region: str,
    repo_url: str,
    key_pair_name: Optional[str],
    security_group: str,
    subnet_tag: Optional[str],
    ami_pattern: Optional[str],
    ami_owner: str,
    iam_instance_profile: Optional[str],
    root_volume_size: int,
    keep_on_stop: bool,
    user_data_file: pathlib.Path,
    extra_tags: tuple[str, ...],
    ami_id: Optional[str],
    restrict_cstates: bool,
    open_file_limit: int,
    disable_smt: bool,
    core_count: Optional[int],
    stabilize_performance: bool,
    disable_grub: bool,
    bare_bones: bool,
    ubuntu_version: str,
) -> None:
    """
    Launch a benchmark EC2 instance suitable for dataloader performance sweeps.
    """
    global REGION_NAME
    REGION_NAME = region
    shutdown_behavior = "stop" if keep_on_stop else "terminate"
    if not key_pair_name and not iam_instance_profile:
        raise click.UsageError(
            "Supply --key-pair-name for SSH or --iam-instance-profile with Session Manager permissions."
        )

    if not keep_on_stop:
        LOGGER.warning(
            "WARNING: --keep-on-stop is NOT set. The instance will be TERMINATED "
            "(permanently destroyed) if shut down from the OS. Pass --keep-on-stop "
            "to preserve the instance on shutdown."
        )

    try:
        user_data = load_user_data(user_data_file)
        user_data = _inject_snippet(
            user_data, f"export REPO_URL={shlex.quote(repo_url)}"
        )

        if bare_bones:
            LOGGER.info(
                "Bare-bones mode: skipping all custom systemd services, GRUB changes, and performance tuning."
            )
            # Inject env var so bootstrap skips service installation
            user_data = _inject_snippet(user_data, "export BARE_BONES=1")

        if restrict_cstates and not disable_grub and not bare_bones:
            LOGGER.info("Injecting GRUB C-State restriction logic into user-data")
            user_data = inject_cstate_setup(user_data)
        elif restrict_cstates and disable_grub:
            LOGGER.info("Skipping C-State boot restrictions (--disable-grub).")

        if open_file_limit > 0:
            LOGGER.info(
                "Injecting open file limit (nofile=%s) configuration into user-data",
                open_file_limit,
            )
            user_data = inject_open_file_limit(user_data, open_file_limit)

        if stabilize_performance and not bare_bones:
            LOGGER.info("Injecting performance stabilization service into user-data")
            user_data = inject_perf_stabilize(user_data)

        is_metal = _is_metal_instance(instance_type)

        if disable_smt and is_metal and not bare_bones:
            LOGGER.info(
                "Bare-metal instance detected (%s); SMT will be disabled via OS-level user-data instead of CpuOptions.",
                instance_type,
            )
            user_data = inject_smt_disable(user_data, include_grub=not disable_grub)
            if disable_grub:
                LOGGER.info(
                    "Skipping GRUB nosmt param (--disable-grub). SMT disabled at runtime only."
                )

        resolved_core_count = core_count
        if disable_smt and not is_metal and not resolved_core_count:
            resolved_core_count = resolve_default_core_count(instance_type)

        config = LaunchConfig(
            instance_name=instance_name,
            instance_type=instance_type,
            key_pair_name=key_pair_name,
            security_group_name=security_group,
            subnet_tag=subnet_tag,
            ami_pattern=ami_pattern,
            ami_owner=ami_owner,
            iam_instance_profile=iam_instance_profile,
            root_volume_size=root_volume_size,
            shutdown_behavior=shutdown_behavior,
            user_data=user_data,
            tags=collect_tags(extra_tags, instance_name),
            ami_id=ami_id,
            restrict_cstates=restrict_cstates,
            open_file_limit=open_file_limit,
            disable_smt=disable_smt,
            core_count=resolved_core_count,
            stabilize_performance=stabilize_performance,
            ubuntu_version=ubuntu_version,
        )

        instance_id, private_ip = create_instance(config)
        output_launch_details(
            instance_name, instance_id, private_ip, key_pair_name, iam_instance_profile
        )
        if (
            not disable_grub
            and not bare_bones
            and (restrict_cstates or (disable_smt and is_metal))
        ):
            LOGGER.info("NOTE: Reboot the instance to apply the GRUB changes.")
    except LaunchError as exc:
        LOGGER.error("Failed to launch instance: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
