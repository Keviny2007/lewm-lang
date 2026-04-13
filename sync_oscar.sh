#!/bin/bash
# ============================================
# Sync Script for Brown Oscar Cluster
# ============================================
# Usage:
#   ./sync_oscar.sh push    - Push code to Oscar
#   ./sync_oscar.sh pull    - Pull results from Oscar
#   ./sync_oscar.sh status  - Check job status
# ============================================

OSCAR_USER="kyang128"
OSCAR_HOST="ssh.ccv.brown.edu"
REMOTE_DIR="scratch/lewm"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

case "$1" in
    push)
        echo -e "${GREEN}Pushing code to Oscar...${NC}"
        rsync -avz --progress \
            --exclude '.venv/' \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            --exclude '.git/' \
            --exclude 'checkpoints/' \
            --exclude 'results/' \
            --exclude 'logs/' \
            --exclude '.pytest_cache/' \
            --exclude '.DS_Store' \
            --exclude '.env' \
            --exclude '*.egg-info/' \
            --exclude '*.h5' \
            --exclude '*.pdf' \
            ./ $OSCAR_USER@$OSCAR_HOST:$REMOTE_DIR/
        echo -e "${GREEN}Done!${NC}"
        ;;

    pull)
        echo -e "${GREEN}Pulling results, checkpoints, and logs from Oscar...${NC}"
        rsync -avz --progress \
            $OSCAR_USER@$OSCAR_HOST:"$REMOTE_DIR/{results,checkpoints,logs}" \
            ./ 2>/dev/null || echo "Some directories may not exist yet"
        echo -e "${GREEN}Done!${NC}"
        ;;

    status)
        echo -e "${GREEN}Checking job status on Oscar...${NC}"
        ssh $OSCAR_USER@$OSCAR_HOST "squeue -u \$USER"
        ;;

    ssh)
        echo -e "${GREEN}Connecting to Oscar...${NC}"
        ssh $OSCAR_USER@$OSCAR_HOST
        ;;

    submit)
        if [ -z "$2" ]; then
            echo -e "${RED}Usage: $0 submit <script_name>${NC}"
            echo "Available scripts in slurm/:"
            ssh $OSCAR_USER@$OSCAR_HOST "ls $REMOTE_DIR/slurm/*.sh 2>/dev/null | xargs -n1 basename | sed 's/.sh//'"
            exit 1
        fi
        echo -e "${GREEN}Submitting $2 job...${NC}"
        ssh $OSCAR_USER@$OSCAR_HOST "cd $REMOTE_DIR && sbatch slurm/$2.sh"
        ;;

    *)
        echo "Oscar Sync Script for LeWM-VLA"
        echo ""
        echo "Usage: $0 {push|pull|status|ssh|submit}"
        echo ""
        echo "Commands:"
        echo "  push              Push code to Oscar"
        echo "  pull              Pull results/checkpoints/logs from Oscar"
        echo "  status            Check SLURM job status"
        echo "  ssh               SSH into Oscar"
        echo "  submit <script>   Submit a SLURM job"
        exit 1
        ;;
esac
