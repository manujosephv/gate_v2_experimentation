git config --global credential.helper "/bin/bash /git_creds.sh"
echo '#!/bin/bash' > /git_creds.sh
echo "sleep 1" >> /git_creds.sh
echo "echo username=manujosephv" >> /git_creds.sh
echo "echo password=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" >> /git_creds.sh