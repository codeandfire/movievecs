#!/usr/bin/env bash

moviename="$1"

# search for this movie name
matches=$(grep -i "$moviename" 'netflix/movie_titles.csv')

if [[ -z $matches ]]; then
	echo "$moviename not found."
	exit
fi

# display matches
echo "ID,Year,Title"
echo "$matches"

# number of matches found
num_matches=$(echo "$matches" | wc -l)

if [[ $num_matches = 1 ]]; then

	# extract the movie id and confirm
	movieid="${matches%%,*}"
	read -p "Confirm ID $movieid? [y/n] "

	if [[ $REPLY = n ]]; then
		exit
	fi
else
	# ask for manual entry of the id
	read -p 'Manually enter ID: ' movieid
fi

# the combined_data_<>.txt file containing ratings for this id
if (( $movieid < 4500 )); then
	fileid='1'
elif (( $movieid < 9211 )); then
	fileid='2'
elif (( $movieid < 13368 )); then
	fileid='3'
else
	fileid='4'
fi

# if this id is the last id in its file
if [[ $movieid = 4499 || $movieid = 9210 || $movieid = 13367 || $movieid = 17770 ]]; then

	# extract lines from the line '<id>:' up to the end of the file
	pattern="/^$movieid:$/,//p"
	striplines='1,1d'
else
	# extract lines from the line '<id>:' up to '<id + 1>:'
	nextmovieid=$(( $movieid + 1 ))
	pattern="/^$movieid:$/,/^$nextmovieid:$/p"
	striplines='1,1d; $d'
fi

# file in which extracted ratings will be stored
filename="${moviename// /_}_$movieid.txt"

# extract ratings
sed -n "$pattern" "netflix/combined_data_$fileid.txt" | sed "$striplines" > "$filename"

echo "Ratings saved to $filename"
