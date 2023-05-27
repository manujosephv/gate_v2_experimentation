DIR="GATE_v2"
if [ -d "$DIR" ]; then
  ### Take action if $DIR exists ###
  echo "Pulling Latest Changes"
  cd $DIR
  git pull origin
else
  ###  Control will jump here if $DIR does NOT exists ###
  echo "Error: ${DIR} not found. Cloning it from source."
  git clone https://github.com/manujosephv/GATE_v2
  cd $DIR
fi

echo "Installing PyTorch Tabular with GATE"
pip install .
cd ..
echo "Installing Other Requirements"
pip install -r requirements.txt

git config --global user.email "manujosephv@gmail.com"
git config --global user.name "Manu Joseph"
