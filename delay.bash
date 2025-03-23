# To run script:
# TEMP_FILE=$(mktemp) && curl -H 'Cache-Control: no-cache' https://gist.githubusercontent.com/aidando73/23bbfa534a01ebefc4b2ef505ca6b464/raw/stop_pod_after_delay.bash -o "$TEMP_FILE" && bash "$TEMP_FILE" && rm -f "$TEMP_FILE"

# Exit immediately if a command exits with a non-zero status
set -e

# On amazon linux
sudo yum -y install tmux
wget -qO- cli.runpod.net | sudo bash

# Prompt for API key
if [ -z "$api_key" ]; then
  echo "Visit https://www.runpod.io/console/user/settings to generate an API key"
  read -p "Enter your RunPod API key: " api_key
  if [ -z "$api_key" ]; then
    echo "API key is required. Exiting."
    exit 1
  fi
fi

# Configure API key
echo "Configuring API key..."
runpodctl config --apiKey "$api_key"
  
# Prompt for pod_id if not provided
if [ -z "$pod_id" ]; then
  read -p "Enter your pod ID: " pod_id
fi

# Prompt for duration if not provided
if [ -z "$duration" ]; then
  read -p "Enter duration (e.g. 90m, 2h, 30s): " duration
  if [ -z "$duration" ]; then
    echo "Duration is required. Exiting."
    exit 1
  fi
fi

# Verify pod exists before proceeding
echo "Verifying pod exists..."
runpodctl get pod $pod_id

# Ask user to confirm this is the correct pod
read -p "Is this the correct pod? (y/n): " confirm
if [[ ! "$confirm" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    echo "Operation cancelled by user."
    exit 1
fi

end_time=$(($(date +%s) + $(echo $duration | sed 's/h/*3600/;s/m/*60/;s/s//;s/$/*1/' | bc)))
while [ $(date +%s) -lt $end_time ]; do
    remaining=$((end_time - $(date +%s)))
    printf "\rTime remaining: %02d:%02d:%02d" $((remaining/3600)) $((remaining%3600/60)) $((remaining%60))
    sleep 1
done

echo -e "\nStopping pod..." 
runpodctl stop pod $pod_id