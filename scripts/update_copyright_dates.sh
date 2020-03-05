#!/bin/bash


#Copyright (c) 2016-0, The Linux Foundation. All rights reserved

for file in `git grep Copyright | grep "Linux Foundation" | cut -d: -f1 | uniq`
do
	lastyear=`git log --format="%ad" -1 --date=short $file | cut -d- -f1`
	firstyear=`(grep Copyright ${file} | grep "Linux Foundation" | sed -e 's/.*Copyright (c) \(20[0-9]*\)[^0-9].*/\1/') < $file`
	echo $file ${firstyear}-${lastyear}
	if [ "${firstyear}" = "${lastyear}" ]
	then
		yearstr=${firstyear}
	else
		yearstr=${firstyear}-${lastyear}
	fi
	sed -i.copyright.update.bak -e 's/\(.*Copyright (c) \)\(20[0-9]*\)\(-20..\)*[ ,]*\(The Linux Foundation.*\)$/\1'${yearstr}', \4/' $file 
done


