for ref in M009 W016
do
  for actor in M003 M027 W019 W037
  do
  ./test_reference_driven.sh ${actor}_neutral ${ref}_happy paired ${actor}_happy 
  done
done