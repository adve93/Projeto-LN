#!/bin/zsh

mkdir -p compiled images

rm -f ./compiled/*.fst ./images/*.pdf

# ############ Compile source transducers ############
for i in sources/*.txt tests/*.txt; do
	echo "Compiling: $i"
    fstcompile --isymbols=syms.txt --osymbols=syms.txt $i | fstarcsort > compiled/$(basename $i ".txt").fst
done

# ############ CORE OF THE PROJECT  ############

#Concating mmm2mm.fst, transducer that translates month in a 3 letter format ex. SEP into a number format ex. 09, with word 2 word
#transducer that simply accepts everyword.
fstconcat ./compiled/mmm2mm.fst ./compiled/word2word.fst > ./compiled/mix2numerical.fst

#Inverse the input and output of the english to portuguese translator to create a portuguese to english translator
fstinvert ./compiled/en2pt.fst > ./compiled/pt2en.fst

#Concating month.fst - bar2coma.fst - day.fst - bar2come.fst - year.fst
fstconcat ./compiled/month.fst ./compiled/bar2coma.fst > ./compiled/aux1.fst
fstconcat ./compiled/aux1.fst ./compiled/day.fst > ./compiled/aux2.fst
fstconcat ./compiled/aux2.fst ./compiled/bar2coma.fst > ./compiled/aux3.fst
fstconcat ./compiled/aux3.fst ./compiled/year.fst > ./compiled/datenum2text.fst

#Composing fst to create mix2text.fst
fstcompose ./compiled/pt2en.fst ./compiled/mix2numerical.fst > ./compiled/aux4.fst
fstconcat  ./compiled/aux4.fst ./compiled/word2word.fst > ./compiled/aux5.fst
fstunion ./compiled/mix2numerical.fst ./compiled/aux5.fst > ./compiled/aux6.fst
fstcompose ./compiled/aux6.fst ./compiled/datenum2text.fst > ./compiled/aux7.fst
fstrmepsilon ./compiled/aux7.fst > ./compiled/mix2text.fst

#Creating date2text with a union
fstunion ./compiled/mix2text.fst ./compiled/datenum2text.fst > ./compiled/date2text.fst

# ############ generate PDFs  ############
echo "Starting to generate PDFs"
for i in compiled/*.fst; do
	echo "Creating image: images/$(basename $i '.fst').pdf"
   fstdraw --portrait --isymbols=syms.txt --osymbols=syms.txt $i | dot -Tpdf > images/$(basename $i '.fst').pdf
done


fst2word() {
	awk '{if(NF>=3){printf("%s",$3)}}END{printf("\n")}'
}

trans=mix2text.fst
echo -e "\nTesting mix2text"
for w in MAY/15/2020 MAI/15/2020; do
    res=$(python3 ./scripts/word2fst.py $w | fstcompile --isymbols=syms.txt --osymbols=syms.txt | fstarcsort |
                       fstcompose - compiled/$trans | fstshortestpath | fstproject --project_output=true |
                       fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=./scripts/syms-out.txt | fst2word)
    echo "$w = $res"

done

echo "\nThe end"
