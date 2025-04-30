#!/bin/bash

IQS=./data/canal.txt
CLI=elv
SAVETO=./video_tags

if [ ! -d "$SAVETO" ]; then
    mkdir "$SAVETO"
    if [ $? -ne 0 ]; then
        echo "Failed to create directory $SAVETO."
        exit 1
    fi
fi
echo "Directory $SAVETO is ready."

# export SECRET=0x.. or read from user input
if [ -z "$SECRET" ]; then
    echo "Enter your account private key:"
    read -r SECRET
else
    echo "Read private key from env variable."
fi

names=()
while IFS= read -r line || [[ -n "$line" ]]; do
    names+=("$line")
done < ${IQS}

# Loop through the list and create directories if folder not exists
# Read from the file, ensuring a newline at the end
for name in "${names[@]}"; do
    dir="$SAVETO/$name"
    if [ ! -d "$dir" ]; then
        mkdir "$dir"
        if [ $? -ne 0 ]; then
            echo "Failed to create directory $dir."
            continue
        fi
    fi

    # Proceed with file download
    $CLI files download --qid "$name" /video_tags "$dir" --secret "${SECRET}" --progress --decryption-mode none
    if [ $? -eq 0 ]; then
        echo "video_tags '$name' downloaded successfully."
    else
        echo "Failed to download video_tags for '$name'."
    fi
done

# cat ../_iqs.txt | xargs -I {} elv files download --qid {} /video_tags ../video_tags/{} --secret ${secret} --progress  --decryption-mode none