git config --global credential.helper "/bin/bash /git_creds.sh"
echo '#!/bin/bash' > /git_creds.sh
echo "sleep 1" >> /git_creds.sh
echo "echo username=manujosephv" >> /git_creds.sh
echo "echo password=<<PAT>>" >> /git_creds.sh