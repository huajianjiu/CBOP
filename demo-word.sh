if [ ! -f "text9" ]
then
  wget -c http://mattmahoney.net/dc/enwik9.zip -P "${DATADIR}"
  unzip "${DATADIR}/enwik9.zip" -d "${DATADIR}"
  perl wikifil.pl "${DATADIR}/enwik9" > "${DATADIR}"/text9
fi

time ./pp_word2vec -train text9 -output vectors_ppdb2.bin -cbow 1 -size 200 -window 8 -negative 25 -sample 1e-4 -threads 20 -binary 1 -iter 15 -dropout 1
./compute-accuracy vectors_ppdb2.bin < questions-words.txt
