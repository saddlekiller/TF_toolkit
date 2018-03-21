basepath="F:/github/TF_toolkit/data/FACE/files"
for i in $(ls $basepath); do
  python tools.py $basepath $i
done
