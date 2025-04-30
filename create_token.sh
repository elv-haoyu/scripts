#!/bin/bash

# Function to display help
function display_help() {
    echo "Usage: $0 <qid> <pkey> [token_flag]"
    echo
    echo "Arguments:"
    echo "  qid          The content object ID"
    echo "  pkey         The private key used to authenticate the request"
    echo "  token_flag   (Optional) Specify --reenc, --update, or --state-channel for token type"
    echo
    echo "Example:"
    echo "  $0 iq__example123456 your_private_key --update"
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
    exit 0
fi

# Check if correct number of arguments is provided (at least 2 arguments)
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    display_help
    exit 1
fi

# Arguments
qid=$1  # qid
pkey=$2 # pkey
token_flag=${3:-""}  # The optional third argument (token flag like --reenc, --update, or --state-channel)

AUTH_TOKEN=$(elv content token create ${qid} ${token_flag} --secret ${pkey} | jq -r .bearer)

# Check if token generation was successful
if [ -z "$AUTH_TOKEN" ]; then
    echo "Failed to generate AUTH_TOKEN."
    exit 1
fi

# Append the token to token.txt
echo $AUTH_TOKEN >> token.txt

# Print out the qid and AUTH_TOKEN
echo "qid (qid): $qid"
echo "AUTH_TOKEN: $AUTH_TOKEN"

# Notify the user
echo "AUTH_TOKEN successfully written to token.txt."
