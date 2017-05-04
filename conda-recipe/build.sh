# Install python modules
mkdir -p ${PREFIX}/watersheds
cp watersheds/*.py ${PREFIX}/watersheds
echo "${PREFIX}" > ${PREFIX}/lib/python2.7/site-packages/watersheds.pth
python -m compileall ${PREFIX}/watersheds
