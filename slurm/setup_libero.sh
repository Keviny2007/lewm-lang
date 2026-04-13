#!/bin/bash
#SBATCH --job-name=lewm-setup-libero
#SBATCH --output=logs/setup_libero_%j.out
#SBATCH --error=logs/setup_libero_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

set -euo pipefail

cd ~/scratch/lewm
mkdir -p logs

echo "=== Installing LIBERO into existing .venv ==="

module load python/3.11.11-5e66
module load cuda/12.9.0-cinr
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

# mujoco binary (needed before robosuite/libero)
uv pip install mujoco

# robosuite — LIBERO depends on a specific version
uv pip install robosuite==1.4.1

# LIBERO from source (pip package may be outdated)
LIBERO_DIR="$HOME/scratch/LIBERO"
if [ ! -d "$LIBERO_DIR" ]; then
    echo "Cloning LIBERO..."
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi
cd "$LIBERO_DIR"
git pull

# clean up broken editable install
SITE="$HOME/scratch/lewm/.venv/lib/python3.11/site-packages"
pip uninstall -y libero 2>/dev/null || true
rm -f "$SITE/__editable__"*libero*
rm -rf "$SITE/libero" "$SITE/libero-"*.dist-info
echo "Cleaned site-packages of old libero artifacts"
ls "$SITE" | grep -i libero || echo "(none remaining)"

# LIBERO's setup.py produces an empty wheel (packaging bug).
# Instead, add the repo root to a .pth file so Python finds the package.
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "$LIBERO_DIR" > "$SITE/libero.pth"
echo "Added $LIBERO_DIR to $SITE/libero.pth"

# reset LIBERO path config to point at the cloned repo
python -c "from libero.libero import set_libero_default_path; set_libero_default_path('$LIBERO_DIR/libero/libero')"

# LIBERO env dependencies (setup.py has empty install_requires)
uv pip install bddl easydict cloudpickle termcolor imageio gym
cd ~/scratch/lewm

# h5py for goal image extraction (likely already installed, but ensure it)
uv pip install h5py

echo ""
echo "=== Verifying ==="
python -c "import mujoco; print(f'mujoco {mujoco.__version__}')"
python -c "import robosuite; print(f'robosuite {robosuite.__version__}')"
python -c "import libero; print('libero OK')"
python -c "from libero.libero import benchmark; print(f'benchmark suites: {list(benchmark.get_benchmark_dict().keys())}')"
python -c "from libero.libero.envs import OffScreenRenderEnv; print('OffScreenRenderEnv OK')"

echo ""
echo "=== LIBERO setup complete ==="
