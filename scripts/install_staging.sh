#!/bin/bash

# Section 1: Build/Install
# This section is for first-time setup and installations.

install_dependencies() {
    # Function to install packages on macOS
    install_mac() {
        which brew > /dev/null
        if [ $? -ne 0 ]; then
            echo "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        echo "Updating Homebrew packages..."
        brew update
        echo "Installing required packages..."
        brew install make llvm curl libssl protobuf tmux jq
    }

    # Function to install packages on Ubuntu/Debian
    install_ubuntu() {
        echo "Updating system packages..."
        sudo apt update
        echo "Installing required packages..."
        sudo apt install --assume-yes make build-essential git clang curl libssl-dev llvm libudev-dev protobuf-compiler tmux
    }

    # Detect OS and call the appropriate function
    if [[ "$OSTYPE" == "darwin"* ]]; then
        install_mac
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        install_ubuntu
    else
        echo "Unsupported operating system."
        exit 1
    fi

    # Install rust and cargo
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

    # Update your shell's source to include Cargo's path
    source "$HOME/.cargo/env"
}

# Call install_dependencies only if it's the first time running the script
if [ ! -f ".dependencies_installed" ]; then
    install_dependencies
    touch .dependencies_installed
fi


# Section 2: Test/Run
# This section is for running and testing the setup.

# Create a coldkey for the owner role
wallet=${1:-owner}
wallet_path=${2:-~/.bittensor/wallets}

# Logic for setting up and running the environment
setup_environment() {
    # todo(alex): Clean up cd directory flow here
    # Clone subtensor and enter the directory
    if [ ! -d "subtensor" ]; then
        git clone https://github.com/opentensor/subtensor.git
    fi
    cd subtensor
    git pull

    # Update to the nightly version of rust
    ./scripts/init.sh

    cd bittensor-subnet-template

    # Install the bittensor-subnet-template python package
    python -m pip install -e .

    # Create and set up wallets
    # This section can be skipped if wallets are already set up
    if [ ! -f ".wallets_setup" ]; then
        btcli wallet new_coldkey --wallet.name owner --no-use-password --n-words 12 --wallet-path $wallet_path
        btcli wallet new_hotkey --wallet.name owner --wallet.hotkey default --n-words 12 --wallet-path $wallet_path

        btcli wallet new_coldkey --wallet.name miner --no-use-password --n-words 12 --wallet-path $wallet_path
        btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default --n-words 12 --wallet-path $wallet_path

        btcli wallet new_coldkey --wallet.name validator --no-use-password --n-words 12 --wallet-path $wallet_path
        btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default --n-words 12 --wallet-path $wallet_path

        touch .wallets_setup
    fi

}

# Call setup_environment every time
setup_environment 

## Setup localnet
# assumes we are in the bittensor-subnet-template/ directory
# Initialize your local subtensor chain in development mode. This command will set up and run a local subtensor network.
pushd subtensor > /dev/null

# Start a new tmux session and create a new pane, but do not switch to it
echo "FEATURES='pow-faucet runtime-benchmarks' BT_DEFAULT_TOKEN_WALLET=$(cat $wallet_path/owner/coldkeypub.txt |  jq -r ".ss58Address") bash scripts/localnet.sh" >> setup_and_run.sh
chmod +x setup_and_run.sh
tmux new-session -d -s localnet -n 'localnet'
tmux send-keys -t localnet 'bash setup_and_run.sh' C-m

# Notify the user
echo ">> localnet.sh is running in a detached tmux session named 'localnet'"
echo ">> You can attach to this session with: tmux attach-session -t localnet"

# Configure btcli
echo "y" | btcli config set --subtensor.network ws://127.0.0.1:9946 --wallet.path $wallet_path

# Register a subnet (this needs to be run each time we start a new local chain)
btcli subnet create --wallet.name owner --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946 --no_prompt --wallet-path $wallet_path

# Transfer tokens to miner and validator coldkeys
export BT_MINER_TOKEN_WALLET=$(cat $wallet_path/miner/coldkeypub.txt |  jq -r ".ss58Address")
export BT_VALIDATOR_TOKEN_WALLET=$(cat $wallet_path/validator/coldkeypub.txt | jq -r ".ss58Address")

# Give funds to miner and validator
btcli wallet transfer --wallet.name owner --dest $BT_MINER_TOKEN_WALLET --amount 1000 --no_prompt
btcli wallet transfer --wallet.name owner --dest $BT_VALIDATOR_TOKEN_WALLET --amount 1000 --no_prompt

# Register wallet hotkeys to subnet
btcli subnet register --wallet.name miner --netuid 1 --wallet.hotkey default --no_prompt
btcli subnet register --wallet.name validator --netuid 1 --wallet.hotkey default --no_prompt

# Add stake to the validator
btcli stake add --wallet.name validator --wallet.hotkey default --amount 1000 --no_prompt

# Ensure both the miner and validator keys are successfully registered.
btcli subnet list --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli wallet overview --wallet.name validator
btcli wallet overview --wallet.name miner

run_miner_command="python neurons/miner.py --netuid 1 --wallet.path $wallet_path --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name miner --wallet.hotkey default --logging.debug"
run_validator_command="python neurons/validator.py --netuid 1 --wallet.path $wallet_path --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name3 validator --wallet.hotkey default --logging.debug"

# Check if inside a tmux session
if [ -z "$TMUX" ]; then
    # Start a new tmux session and run the miner in the first pane
    tmux new-session -d -s bittensor -n 'miner' $run_miner_command
    
    # Split the window and run the validator in the new pane
    tmux split-window -h -t bittensor:miner $run_validator_command
    
    # Attach to the new tmux session
    tmux attach-session -t bittensor
else
    # If already in a tmux session, create two panes in the current window
    tmux split-window -h $run_miner_command
    tmux split-window -v -t 0 $run_validator_command
fi
