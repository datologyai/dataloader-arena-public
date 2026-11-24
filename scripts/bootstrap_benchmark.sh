#!/usr/bin/env bash
set -euxo pipefail

MOUNT_POINT="/mnt/local"
UBUNTU_USER="ubuntu"
BENCHMARK_ROOT="/opt/benchmark"
REPO_URL="${REPO_URL:-https://github.com/datologyai/dataloader-arena-public.git}"

log() {
  echo "[$(date --iso-8601=seconds)] $*"
}

configure_timezone() {
  if [ -z "${TIME_ZONE:-}" ]; then
    return 0
  fi
  log "Setting timezone to ${TIME_ZONE}"
  timedatectl set-timezone "${TIME_ZONE}"
}

update_packages() {
  log "Updating apt package index"
  export DEBIAN_FRONTEND=noninteractive
  local attempt
  for attempt in 1 2 3; do
    if apt-get update -y; then
      break
    fi
    log "apt-get update failed (attempt ${attempt}/3), cleaning lists and retrying in 10s..."
    rm -rf /var/lib/apt/lists/*
    sleep 10
  done
  apt-get upgrade -y
}

install_base_packages() {
  log "Installing base packages"
  apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    git \
    gnupg \
    lsb-release \
    software-properties-common \
    unzip \
    tmux \
    build-essential \
    pkg-config \
    python3 \
    python3-venv \
    libffi-dev \
    zlib1g-dev \
    cmake \
    mdadm \
    nvme-cli
}

install_awscli() {
  log "Installing AWS CLI v2"
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -qo /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install --update
  rm -rf /tmp/awscliv2.zip /tmp/aws
  aws --version


}

install_docker() {
  log "Installing Docker engine"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg

  OS_CODENAME="$(lsb_release -cs)"
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    ${OS_CODENAME} stable" |
    tee /etc/apt/sources.list.d/docker.list >/dev/null

  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  systemctl enable docker
  systemctl start docker
  usermod -aG docker "${UBUNTU_USER}"
}

install_uv() {
  log "Installing uv for Python dependency management"
  sudo -H -u "${UBUNTU_USER}" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

  cat >/etc/profile.d/uv.sh <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
EOF
  chmod 0644 /etc/profile.d/uv.sh

  # Make uv available system-wide for non-login shells (e.g., some SSM sessions)
  if [ -x "/home/${UBUNTU_USER}/.local/bin/uv" ]; then
    ln -sf "/home/${UBUNTU_USER}/.local/bin/uv" /usr/local/bin/uv || true
  fi
}

setup_instance_store_mount() {
  # Deploy a systemd service that discovers, formats-if-needed, and mounts
  # NVMe instance store on every boot. This handles:
  #   - First boot (format + mount)
  #   - Reboot (already formatted, just mount)
  #   - Stop/start (data wiped, needs re-format + mount)
  log "Installing instance-store mount service"

  cat >/usr/local/sbin/mount-instance-store.sh <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

MOUNT_POINT="/mnt/local"
OWNER="ubuntu"

# Find NVMe instance store devices (not EBS).
# Instance store model string contains "Instance Storage".
discover_instance_store_devices() {
  local devices=()
  for dev in /dev/nvme*n1; do
    [ -b "$dev" ] || continue
    model="$(cat "/sys/block/$(basename "$dev")/device/model" 2>/dev/null || true)"
    if [[ "$model" == *"Instance Storage"* ]]; then
      devices+=("$dev")
    fi
  done
  echo "${devices[@]}"
}

devices=( $(discover_instance_store_devices) )

if [ ${#devices[@]} -eq 0 ]; then
  echo "No NVMe instance store devices found; nothing to mount."
  exit 0
fi

mkdir -p "${MOUNT_POINT}"

if [ ${#devices[@]} -eq 1 ]; then
  DEV="${devices[0]}"
  echo "Single instance store device: ${DEV}"
  if ! blkid "${DEV}" >/dev/null 2>&1; then
    echo "No filesystem on ${DEV}; formatting with ext4"
    mkfs.ext4 -F "${DEV}"
  fi
  mount -o defaults,noatime "${DEV}" "${MOUNT_POINT}"
else
  # Multiple instance store devices: assemble as RAID-0 for max throughput
  MD_DEV="/dev/md0"
  echo "Found ${#devices[@]} instance store devices: ${devices[*]}"

  if [ -b "${MD_DEV}" ] && grep -q "active" /proc/mdstat 2>/dev/null; then
    # Array already assembled (reboot case) — just mount it
    echo "RAID array ${MD_DEV} already active; mounting."
    if ! mountpoint -q "${MOUNT_POINT}"; then
      mount -o defaults,noatime "${MD_DEV}" "${MOUNT_POINT}"
    fi
  else
    # Fresh creation (first boot or stop/start)
    echo "Creating RAID-0 array at ${MD_DEV}"
    mdadm --create "${MD_DEV}" --level=0 --raid-devices=${#devices[@]} \
      "${devices[@]}" --force --run --homehost=any
    if ! blkid "${MD_DEV}" >/dev/null 2>&1; then
      echo "Formatting ${MD_DEV} with ext4"
      mkfs.ext4 -F "${MD_DEV}"
    fi
    mount -o defaults,noatime "${MD_DEV}" "${MOUNT_POINT}"
  fi

  # Write mdadm.conf so the system knows about this array
  mkdir -p /etc/mdadm
  mdadm --detail --scan > /etc/mdadm/mdadm.conf 2>/dev/null || true
  update-initramfs -u 2>/dev/null || true
fi

chown -R "${OWNER}:${OWNER}" "${MOUNT_POINT}"
echo "Instance store mounted at ${MOUNT_POINT}"
SCRIPT

  chmod 0755 /usr/local/sbin/mount-instance-store.sh

  cat >/etc/systemd/system/mount-instance-store.service <<'UNIT'
[Unit]
Description=Format and mount NVMe instance store
After=local-fs.target
Before=cloud-final.service

[Service]
Type=oneshot
RemainAfterExit=yes
TimeoutStartSec=60
TimeoutStopSec=30
ExecStart=/usr/local/sbin/mount-instance-store.sh
ExecStop=/bin/bash -c 'umount /mnt/local 2>/dev/null; [ -b /dev/md0 ] && mdadm --stop /dev/md0 2>/dev/null; true'

[Install]
WantedBy=multi-user.target
UNIT

  systemctl daemon-reload
  systemctl enable mount-instance-store.service

  # Also run it now for the first boot
  systemctl start mount-instance-store.service

  log "Instance store mount service installed and started"
}

clone_repo_to_nvme() {
  local REPO_DIR="${MOUNT_POINT}/dataloader-arena"
  local SYMLINK="/home/${UBUNTU_USER}/dataloader-arena-nvme"

  # Try the initial clone if NVMe is mounted (best-effort, will likely
  # fail during cloud-init due to missing SSH key)
  if mountpoint -q "${MOUNT_POINT}" 2>/dev/null; then
    log "Cloning dataloader-arena to NVMe at ${REPO_DIR}"
    if [ ! -d "${REPO_DIR}/.git" ]; then
      sudo -H -u "${UBUNTU_USER}" bash -c "
        export GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=accept-new'
        git clone '${REPO_URL}' '${REPO_DIR}'
      " && {
        ln -sfn "${REPO_DIR}" "${SYMLINK}"
        chown -h "${UBUNTU_USER}:${UBUNTU_USER}" "${SYMLINK}"
      } || log "Git clone failed (SSH key may not be forwarded during cloud-init). Will retry on first SSH login."
    else
      log "Repo already exists at ${REPO_DIR}; fetching latest"
      sudo -H -u "${UBUNTU_USER}" bash -c "cd '${REPO_DIR}' && git fetch --all --tags" || true
      ln -sfn "${REPO_DIR}" "${SYMLINK}"
      chown -h "${UBUNTU_USER}:${UBUNTU_USER}" "${SYMLINK}"
    fi
  else
    log "NVMe not mounted at ${MOUNT_POINT}; skipping initial clone"
  fi

  # Always install the helper script, systemd service, and login hook
  # regardless of whether the initial clone succeeded.
  # Used by both the systemd service (boot) and the login hook (SSH).
  cat >/usr/local/sbin/ensure-nvme-repo.sh <<'ENSURESCRIPT'
#!/usr/bin/env bash
set -euo pipefail

[ ! -f /etc/default/dataloader-arena ] || source /etc/default/dataloader-arena
REPO_URL="${REPO_URL:-https://github.com/datologyai/dataloader-arena-public.git}"
MOUNT_POINT="/mnt/local"
REPO_DIR="${MOUNT_POINT}/dataloader-arena"
SYMLINK="/home/ubuntu/dataloader-arena-nvme"
LOCKFILE="/tmp/.nvme-repo-clone.lock"

# Bail if NVMe isn't mounted
mountpoint -q "${MOUNT_POINT}" 2>/dev/null || exit 0

# Repo already there — just fix symlink
if [ -d "${REPO_DIR}/.git" ]; then
    ln -sfn "${REPO_DIR}" "${SYMLINK}"
    chown -h ubuntu:ubuntu "${SYMLINK}"
    exit 0
fi

# Avoid multiple concurrent clones
exec 9>"${LOCKFILE}"
flock -n 9 || { echo "Clone already in progress."; exit 0; }

echo "Cloning dataloader-arena to NVMe..."
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"
git clone "${REPO_URL}" "${REPO_DIR}" || {
    echo "Clone failed."
    exit 0
}
chown -R ubuntu:ubuntu "${REPO_DIR}"
ln -sfn "${REPO_DIR}" "${SYMLINK}"
chown -h ubuntu:ubuntu "${SYMLINK}"
echo "dataloader-arena cloned to ${REPO_DIR}"
ENSURESCRIPT
  chmod 0755 /usr/local/sbin/ensure-nvme-repo.sh

  # Systemd service: runs on boot after NVMe is mounted.
  # Handles reboot (repo exists, just fix symlink) and stop/start (repo gone, re-clone).
  cat >/etc/systemd/system/nvme-repo-symlink.service <<'SYMLINKUNIT'
[Unit]
Description=Ensure dataloader-arena repo on NVMe
After=mount-instance-store.service
Requires=mount-instance-store.service

[Service]
Type=oneshot
RemainAfterExit=yes
TimeoutStartSec=60
User=ubuntu
ExecStart=/usr/local/sbin/ensure-nvme-repo.sh

[Install]
WantedBy=multi-user.target
SYMLINKUNIT

  systemctl daemon-reload
  systemctl enable nvme-repo-symlink.service

  # Login hook: if user SSHs in and repo is missing, clone in background.
  # This catches the case where systemd clone failed (no SSH key at boot)
  # but the user now has agent forwarding.
  cat >/etc/profile.d/nvme-repo-check.sh <<'LOGINHOOK'
if mountpoint -q /mnt/local 2>/dev/null && [ ! -d /mnt/local/dataloader-arena/.git ]; then
    echo -e "\033[1;33m▸ dataloader-arena not found on NVMe — cloning in background...\033[0m"
    nohup /usr/local/sbin/ensure-nvme-repo.sh >/tmp/nvme-repo-clone.log 2>&1 &
    echo -e "  Follow progress: tail -f /tmp/nvme-repo-clone.log"
fi
LOGINHOOK
  chmod 0644 /etc/profile.d/nvme-repo-check.sh

  log "Repo cloned to NVMe; symlink at ${SYMLINK}"
}

install_rust() {
  log "Installing Rust (root + ${UBUNTU_USER})"

  PROFILE="${RUST_PROFILE:-default}"     # includes rustc, cargo, rust-std
  TOOLCHAIN="${RUST_TOOLCHAIN:-stable}"

  export DEBIAN_FRONTEND=noninteractive

  # Avoid conflicts with distro/snap Rust
  apt-get purge -y rustc cargo || true
  if command -v snap >/dev/null 2>&1; then
    snap remove rustup --purge >/dev/null 2>&1 || true
    snap remove rust   --purge >/dev/null 2>&1 || true
  fi

  # Native deps (harmless if already present)
  apt-get install -y curl ca-certificates build-essential pkg-config libssl-dev

  # Ensure $HOME/.cargo/bin appears on PATH for login shells
  cat >/etc/profile.d/rust.sh <<'EOF'
# Rust via rustup
export PATH="$HOME/.cargo/bin:$PATH"
# (Optional) auto-source env if present (safe no-op if missing)
if [ -f "$HOME/.cargo/env" ]; then . "$HOME/.cargo/env"; fi
EOF
  chmod 0644 /etc/profile.d/rust.sh

  # ---------- root install ----------
  if ! [ -x /root/.cargo/bin/rustup ]; then
    curl -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile "$PROFILE" --default-toolchain "$TOOLCHAIN"
  fi
  /root/.cargo/bin/rustup set profile "$PROFILE"
  /root/.cargo/bin/rustup toolchain install "$TOOLCHAIN" --profile "$PROFILE" -c rustfmt -c clippy
  /root/.cargo/bin/rustup default "$TOOLCHAIN"
  /root/.cargo/bin/rustup update "$TOOLCHAIN" || true

  # ---------- UBUNTU_USER install ----------
  if id -u "${UBUNTU_USER}" >/dev/null 2>&1; then
    sudo -H -u "${UBUNTU_USER}" bash -lc '
      set -euo pipefail
      PROFILE="${RUST_PROFILE:-'"$PROFILE"'}"
      TOOLCHAIN="${RUST_TOOLCHAIN:-'"$TOOLCHAIN"'}"
      if ! [ -x "$HOME/.cargo/bin/rustup" ]; then
        curl -sSf https://sh.rustup.rs \
          | sh -s -- -y --profile "$PROFILE" --default-toolchain "$TOOLCHAIN"
      fi
      "$HOME/.cargo/bin/rustup" set profile "$PROFILE"
      "$HOME/.cargo/bin/rustup" toolchain install "$TOOLCHAIN" --profile "$PROFILE" -c rustfmt -c clippy
      "$HOME/.cargo/bin/rustup" default "$TOOLCHAIN"
      "$HOME/.cargo/bin/rustup" update "$TOOLCHAIN" || true
      "$HOME/.cargo/bin/rustc" --version || true
      "$HOME/.cargo/bin/cargo" --version || true
    '
  else
    log "User ${UBUNTU_USER} not found; skipping user Rust install"
  fi

  # ---------- smart wrapper shims (handle shells with no default configured) ----------
  # These prefer the current default toolchain; if none is set, they fall back to 'stable'.
  install -m 0755 /dev/stdin /usr/local/bin/cargo <<'EOF'
#!/usr/bin/env bash
set -e
BIN="${CARGO_HOME:-$HOME/.cargo}/bin"
if "$BIN/rustup" show active-toolchain >/dev/null 2>&1; then
  exec "$BIN/cargo" "$@"
else
  exec "$BIN/rustup" run stable cargo "$@"
fi
EOF

  install -m 0755 /dev/stdin /usr/local/bin/rustc <<'EOF'
#!/usr/bin/env bash
set -e
BIN="${CARGO_HOME:-$HOME/.cargo}/bin"
if "$BIN/rustup" show active-toolchain >/dev/null 2>&1; then
  exec "$BIN/rustc" "$@"
else
  exec "$BIN/rustup" run stable rustc "$@"
fi
EOF

  install -m 0755 /dev/stdin /usr/local/bin/rustup <<'EOF'
#!/usr/bin/env bash
exec "${CARGO_HOME:-$HOME/.cargo}/bin/rustup" "$@"
EOF

  # ---------- belt & suspenders: explicitly ensure defaults now for both users ----------
  /root/.cargo/bin/rustup default "$TOOLCHAIN" || true
  if id -u "${UBUNTU_USER}" >/dev/null 2>&1; then
    sudo -H -u "${UBUNTU_USER}" bash -lc '"$HOME/.cargo/bin/rustup" default "'"$TOOLCHAIN"'"' || true
  fi

  # Log versions (root) for cloud-init-output.log
  /root/.cargo/bin/rustc --version || true
  /root/.cargo/bin/cargo --version || true
  /root/.cargo/bin/rustup --version || true

  log "Rust install complete (root + ${UBUNTU_USER})"
}

install_benchmark_motd() {
  log "Installing dynamic benchmark MOTD"

  # Disable default Ubuntu MOTD noise
  chmod -x /etc/update-motd.d/* 2>/dev/null || true

  cat >/etc/update-motd.d/01-benchmark-status <<'MOTDSCRIPT'
#!/usr/bin/env bash
# Dynamic MOTD for dataloader-arena benchmark instances

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

divider="────────────────────────────────────────────────────────"

printf "\n${BOLD}${CYAN}  dataloader-arena benchmark instance${RESET}\n"
printf "${divider}\n"

# --- System ---
printf "${BOLD}System${RESET}\n"
printf "  %-22s %s\n" "OS:" "$(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
printf "  %-22s %s\n" "Kernel:" "$(uname -r)"

# Instance type from EC2 metadata (IMDSv2)
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 30" 2>/dev/null || true)
if [ -n "$TOKEN" ]; then
  ITYPE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
  IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
  printf "  %-22s %s (%s)\n" "Instance:" "$ITYPE" "$IID"
fi

# CPU (take only the first Model name line to avoid duplicates)
printf "  %-22s %s\n" "CPU:" "$(lscpu | awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
ONLINE=$(nproc)
SOCKETS=$(lscpu | awk -F: '/Socket\(s\)/ {gsub(/^[ \t]+/, "", $2); print $2}')
TPC=$(lscpu | awk -F: '/Thread\(s\) per core/ {gsub(/^[ \t]+/, "", $2); print $2}')
TOTAL_VCPUS=$(lscpu | awk -F: '/^CPU\(s\)/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')
if [ "$TPC" = "1" ]; then
  printf "  %-22s %s physical cores online (%s total, %s sockets, SMT off)\n" "Topology:" "$ONLINE" "$TOTAL_VCPUS" "$SOCKETS"
else
  printf "  %-22s %s vCPUs online (%s total, %s sockets, %s threads/core)\n" "Topology:" "$ONLINE" "$TOTAL_VCPUS" "$SOCKETS" "$TPC"
fi

# Memory
MEM_TOTAL=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
printf "  %-22s %s GiB\n" "Memory:" "$MEM_TOTAL"

printf "\n"

# --- CPU Tuning ---
printf "${BOLD}CPU Tuning${RESET}\n"

# SMT
SMT_STATUS="n/a"
if [ -f /sys/devices/system/cpu/smt/active ]; then
  if [ "$(cat /sys/devices/system/cpu/smt/active)" = "1" ]; then
    SMT_STATUS="${YELLOW}enabled${RESET}"
  else
    SMT_STATUS="${GREEN}disabled${RESET}"
  fi
fi
printf "  %-22s %b\n" "SMT:" "$SMT_STATUS"

# Governor
GOV="n/a"
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
  GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
  if [ "$GOV" = "performance" ]; then
    GOV="${GREEN}performance${RESET}"
  else
    GOV="${YELLOW}${GOV}${RESET}"
  fi
fi
printf "  %-22s %b\n" "CPU Governor:" "$GOV"

# Turbo boost
TURBO="n/a"
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
  if [ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" = "1" ]; then
    TURBO="${GREEN}disabled (Intel)${RESET}"
  else
    TURBO="${YELLOW}enabled (Intel)${RESET}"
  fi
elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
  if [ "$(cat /sys/devices/system/cpu/cpufreq/boost)" = "0" ]; then
    TURBO="${GREEN}disabled (AMD)${RESET}"
  else
    TURBO="${YELLOW}enabled (AMD)${RESET}"
  fi
fi
printf "  %-22s %b\n" "Turbo Boost:" "$TURBO"

# C-States (check GRUB)
CSTATE="n/a"
if grep -q "intel_idle.max_cstate=1" /proc/cmdline 2>/dev/null; then
  CSTATE="${GREEN}restricted (max_cstate=1)${RESET}"
elif grep -q "intel_idle.max_cstate=1" /etc/default/grub 2>/dev/null; then
  CSTATE="${YELLOW}configured, reboot needed${RESET}"
else
  CSTATE="${YELLOW}unrestricted${RESET}"
fi
printf "  %-22s %b\n" "C-States:" "$CSTATE"

# NMI watchdog
NMI="n/a"
if [ -f /proc/sys/kernel/nmi_watchdog ]; then
  if [ "$(cat /proc/sys/kernel/nmi_watchdog)" = "0" ]; then
    NMI="${GREEN}disabled${RESET}"
  else
    NMI="${YELLOW}enabled${RESET}"
  fi
fi
printf "  %-22s %b\n" "NMI Watchdog:" "$NMI"

# NUMA balancing
NUMA="n/a"
if [ -f /proc/sys/kernel/numa_balancing ]; then
  if [ "$(cat /proc/sys/kernel/numa_balancing)" = "0" ]; then
    NUMA="${GREEN}disabled${RESET}"
  else
    NUMA="${YELLOW}enabled${RESET}"
  fi
fi
printf "  %-22s %b\n" "NUMA Balancing:" "$NUMA"

# THP
THP="n/a"
if [ -d /sys/kernel/mm/transparent_hugepage ]; then
  THP_ENABLED=$(cat /sys/kernel/mm/transparent_hugepage/enabled | grep -oP '\[\K[^\]]+')
  THP_DEFRAG=$(cat /sys/kernel/mm/transparent_hugepage/defrag | grep -oP '\[\K[^\]]+')
  THP="${GREEN}${THP_ENABLED}${RESET} (defrag: ${THP_DEFRAG})"
fi
printf "  %-22s %b\n" "THP:" "$THP"

# ASLR
ASLR="n/a"
if [ -f /proc/sys/kernel/randomize_va_space ]; then
  ASLR_VAL=$(cat /proc/sys/kernel/randomize_va_space)
  if [ "$ASLR_VAL" = "0" ]; then
    ASLR="${GREEN}disabled${RESET}"
  else
    ASLR="${YELLOW}enabled (level ${ASLR_VAL})${RESET}"
  fi
fi
printf "  %-22s %b\n" "ASLR:" "$ASLR"

# Scheduler autogroups
AUTOGROUP="n/a"
if [ -f /proc/sys/kernel/sched_autogroup_enabled ]; then
  if [ "$(cat /proc/sys/kernel/sched_autogroup_enabled)" = "0" ]; then
    AUTOGROUP="${GREEN}disabled${RESET}"
  else
    AUTOGROUP="${YELLOW}enabled${RESET}"
  fi
fi
printf "  %-22s %b\n" "Sched Autogroups:" "$AUTOGROUP"

# IRQ balance
IRQBAL="n/a"
if command -v systemctl >/dev/null 2>&1; then
  if systemctl is-active --quiet irqbalance 2>/dev/null; then
    IRQBAL="${YELLOW}running${RESET}"
  else
    IRQBAL="${GREEN}stopped${RESET}"
  fi
fi
printf "  %-22s %b\n" "IRQ Balance:" "$IRQBAL"

# Swappiness
SWAP="n/a"
if [ -f /proc/sys/vm/swappiness ]; then
  SWAP_VAL=$(cat /proc/sys/vm/swappiness)
  if [ "$SWAP_VAL" -le 1 ]; then
    SWAP="${GREEN}${SWAP_VAL}${RESET}"
  else
    SWAP="${YELLOW}${SWAP_VAL}${RESET}"
  fi
fi
printf "  %-22s %b\n" "Swappiness:" "$SWAP"

# Timer migration
TMIG="n/a"
if [ -f /proc/sys/kernel/timer_migration ]; then
  if [ "$(cat /proc/sys/kernel/timer_migration)" = "0" ]; then
    TMIG="${GREEN}disabled${RESET}"
  else
    TMIG="${YELLOW}enabled${RESET}"
  fi
fi
printf "  %-22s %b\n" "Timer Migration:" "$TMIG"

# I/O scheduler (check first NVMe)
IOSCHED="n/a"
for sched in /sys/block/nvme*/queue/scheduler; do
  if [ -f "$sched" ]; then
    IOSCHED=$(cat "$sched" | grep -oP '\[\K[^\]]+')
    if [ "$IOSCHED" = "none" ]; then
      IOSCHED="${GREEN}none${RESET}"
    else
      IOSCHED="${YELLOW}${IOSCHED}${RESET}"
    fi
    break
  fi
done
printf "  %-22s %b\n" "I/O Scheduler:" "$IOSCHED"

# Perf events — count file descriptors pointing at perf_event
PERF_FDS=$(find /proc/[0-9]*/fd -maxdepth 1 -lname 'anon_inode:\[perf_event\]' 2>/dev/null | wc -l)
if [ "$PERF_FDS" -gt 0 ]; then
  PERF_STATUS="${YELLOW}${PERF_FDS} active fd(s)${RESET}"
else
  PERF_STATUS="${GREEN}idle${RESET}"
fi
printf "  %-22s %b\n" "Perf Events:" "$PERF_STATUS"

# Background services
BG_NOISE=""
for svc in apt-daily.timer unattended-upgrades.service; do
  if systemctl is-active --quiet "$svc" 2>/dev/null; then
    BG_NOISE="${BG_NOISE}${svc} "
  fi
done
if [ -n "$BG_NOISE" ]; then
  printf "  %-22s ${YELLOW}%s${RESET}\n" "Noisy Services:" "$BG_NOISE"
else
  printf "  %-22s ${GREEN}%s${RESET}\n" "Noisy Services:" "none"
fi

printf "\n"

# --- Storage ---
printf "${BOLD}Storage${RESET}\n"

MOUNT_POINT="/mnt/local"
if mountpoint -q "${MOUNT_POINT}" 2>/dev/null; then
  DEV=$(findmnt -no SOURCE "${MOUNT_POINT}")
  SIZE=$(df -h "${MOUNT_POINT}" | awk 'NR==2 {print $2}')
  USED=$(df -h "${MOUNT_POINT}" | awk 'NR==2 {print $5}')

  # Check if RAID
  if [[ "$DEV" == /dev/md* ]]; then
    RAID_DEVS=$(cat /proc/mdstat 2>/dev/null | grep -oP 'nvme\S+' | wc -l)
    [ "$RAID_DEVS" -eq 0 ] && RAID_DEVS="?"
    printf "  %-22s ${GREEN}%s${RESET} (%s RAID-0, %sx NVMe, %s used)\n" "Instance Store:" "$MOUNT_POINT" "$SIZE" "$RAID_DEVS" "$USED"
  else
    printf "  %-22s ${GREEN}%s${RESET} (%s, %s used)\n" "Instance Store:" "$MOUNT_POINT" "$SIZE" "$USED"
  fi
else
  printf "  %-22s ${RED}not mounted${RESET}\n" "Instance Store:"
fi

# Open file limit (read configured value, not the MOTD shell's own limit)
NOFILE=$(grep -m1 'soft nofile' /etc/security/limits.d/99-nofile.conf 2>/dev/null | awk '{print $4}')
[ -z "$NOFILE" ] && NOFILE=$(ulimit -n 2>/dev/null || echo "?")
printf "  %-22s %s\n" "Open File Limit:" "$NOFILE"

printf "\n${divider}\n\n"
MOTDSCRIPT

  chmod 0755 /etc/update-motd.d/01-benchmark-status
  log "Benchmark MOTD installed"
}

prepare_directories() {
  log "Preparing benchmark directories in ${BENCHMARK_ROOT}"
  install -d -o "${UBUNTU_USER}" -g "${UBUNTU_USER}" "${BENCHMARK_ROOT}"/{bin,logs,tmp}
}

deploy_helper_scripts() {
  log "Deploying helper scripts"

  cat >"${BENCHMARK_ROOT}/bin/prepare_repo.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

[ ! -f /etc/default/dataloader-arena ] || source /etc/default/dataloader-arena
WORKDIR="${WORKDIR:-/opt/benchmark/workspace}"
REPO_URL="${REPO_URL:-https://github.com/datologyai/dataloader-arena-public.git}"

# Accept new GitHub host key non-interactively for first-time SSH
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"

mkdir -p "${WORKDIR}"

if [ ! -d "${WORKDIR}/dataloader-arena/.git" ]; then
  git clone "${REPO_URL}" "${WORKDIR}/dataloader-arena"
else
  (cd "${WORKDIR}/dataloader-arena" && git fetch --all --tags)
fi

echo "Repository prepared under ${WORKDIR}"
EOF

  chmod 0755 "${BENCHMARK_ROOT}/bin/prepare_repo.sh"

  cat >"${BENCHMARK_ROOT}/bin/prepare_data.sh" <<'PREPARE'
#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${REPO_DIR:-/mnt/local/dataloader-arena}"
exec uv run --project "${REPO_DIR}" --locked --extra data-preparation --inexact python "${REPO_DIR}/scripts/prepare_data.py" "$@"
PREPARE
  chmod 0755 "${BENCHMARK_ROOT}/bin/prepare_data.sh"

  chown -R "${UBUNTU_USER}:${UBUNTU_USER}" "${BENCHMARK_ROOT}"
}

fix_ssm_after_reboot() {
  # On bare-metal instances, the SSM agent often fails to re-register
  # its session channel after reboot. This service restarts the agent
  # 30s after boot to force re-registration.
  log "Installing SSM agent restart workaround for bare-metal reboots"

  cat >/etc/systemd/system/ssm-agent-restart.service <<'SSMUNIT'
[Unit]
Description=Restart SSM agent after boot (bare-metal workaround)
After=multi-user.target
[Service]
Type=oneshot
ExecStartPre=/bin/sleep 30
ExecStart=/bin/bash -c 'if command -v snap >/dev/null 2>&1 && snap list amazon-ssm-agent >/dev/null 2>&1; then snap restart amazon-ssm-agent; else systemctl restart amazon-ssm-agent; fi'
[Install]
WantedBy=multi-user.target
SSMUNIT

  systemctl daemon-reload
  systemctl enable ssm-agent-restart.service
  log "SSM agent restart service installed"
}

main() {
  mkdir -p /etc/default
  declare -p REPO_URL > /etc/default/dataloader-arena
  chmod 0644 /etc/default/dataloader-arena
  configure_timezone
  update_packages
  install_base_packages
  install_awscli
  install_docker
  install_rust
  install_uv

  if [ "${BARE_BONES:-0}" = "1" ]; then
    log "Bare-bones mode: skipping custom systemd services (RAID, stabilize, repo clone, MOTD)"
  else
    setup_instance_store_mount
    clone_repo_to_nvme
    install_benchmark_motd
  fi

  fix_ssm_after_reboot
  prepare_directories
  deploy_helper_scripts
  log "Bootstrap complete"
}

main "$@"
