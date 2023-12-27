mkdir -p csv
rm -f csv/*.csv
META_DATA_FILE="csv/metadata.csv"
echo "name,rawfile,csvfile,label,rate,size" > $META_DATA_FILE
for file in raw/*.TXT; do
	# Extract basename."
	rawfile="$(basename -- $file)"
	name="${rawfile%.TXT}"
	# Generate output CSV file.
	csvfile="${name}.csv"
	output="csv/${name}.csv"
	tail -n +6 $file > $output
	# Append metadata.
	label=${name:0:3}
	rate=200
#	size=$(wc -l $output)
	size=$(sed -n '$=' $output)
	echo "$name,$rawfile,$csvfile,$label,$rate,$size" >> $META_DATA_FILE
done

