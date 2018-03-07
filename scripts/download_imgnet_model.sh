#-----------------------------------------
# SSH: Single Stage Headless Face Detector
# Download script
#-----------------------------------------
#!/bin/bash
url="legacydirs.umiacs.umd.edu/~najibi/download.php"
method_name="vgg"
file_name="imagenet_models.tar.gz"
cur_dir=${PWD##*/}
target_dir="./data/"

if [ $cur_dir = "scripts" ]; then
   target_dir="../data/"
fi

if [ ! -f ${file_name} ]; then
   echo "Downloading ${file_name}..."
   wget "${url}?name=${method_name}" -O "${target_dir}${file_name}"
   echo "Done!"
else
   echo "File already exists!"
fi

echo "Unzipping the file..."
tar -xvzf "${target_dir}${file_name}" -C ${target_dir}

echo "Cleaning up..."
rm "${target_dir}${file_name}"
echo "All done!"



