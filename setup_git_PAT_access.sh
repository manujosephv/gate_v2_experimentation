git config --global credential.helper "/bin/bash /git_creds.sh"
echo '#!/bin/bash' > /git_creds.sh
echo "sleep 1" >> /git_creds.sh
echo "echo username=manujosephv" >> /git_creds.sh
echo "echo password=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" >> /git_creds.sh