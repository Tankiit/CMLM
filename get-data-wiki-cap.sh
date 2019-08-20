# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

NCODES=30000 # number of BPE codes learned

lg=$1  # input language

# data path
MAIN_PATH=$PWD
COCO_PATH=$PWD/data/coco
CAP_PATH=$COCO_PATH/cap
WIKI_PATH=$COCO_PATH/wiki
PROCESSED_PATH=$PWD/data/processed/coco

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast

# Wiki data
WIKI_DUMP_NAME=${lg}wiki-latest-pages-articles.xml.bz2
WIKI_DUMP_LINK=https://dumps.wikimedia.org/${lg}wiki/latest/$WIKI_DUMP_NAME

# Coco caption data
TRAIN_CAP=$CAP_PATH/raw_caption_train.${lg} #copied from image/coco36
# BPE / vocab files
BPE_CODES=$COCO_PATH/codes
VOCAB=$COCO_PATH/vocab

# install tools
./install-tools.sh

# create Wiki paths
mkdir -p $WIKI_PATH/bz2
mkdir -p $WIKI_PATH/txt
mkdir -p $PROCESSED_PATH
mkdir -p $PROCESSED_PATH/wiki
mkdir -p $PROCESSED_PATH/cap

# download Wikipedia dump
echo "Downloading $lg Wikipedia dump from $WIKI_DUMP_LINK ..."
wget -c $WIKI_DUMP_LINK -P $WIKI_PATH/bz2/
echo "Downloaded $WIKI_DUMP_NAME in $WIKI_PATH/bz2/$WIKI_DUMP_NAME"

# extract and tokenize Wiki data
cd $MAIN_PATH
echo "*** Cleaning and tokenizing $lg Wikipedia dump ... ***"
if [ ! -f $WIKI_PATH/txt/$lg.all ]; then
  python $TOOLS_PATH/wikiextractor/WikiExtractor.py $WIKI_PATH/bz2/$WIKI_DUMP_NAME --processes 8 -q -o - \
  | sed "/^\s*\$/d" \
  | grep -v "^<doc id=" \
  | grep -v "</doc>\$" \
  | $TOKENIZE $lg \
  | python $LOWER_REMOVE_ACCENT \
  > $WIKI_PATH/txt/$lg.all
fi
echo "*** Tokenized (+ lowercase + accent-removal) $lg Wikipedia dump to $WIKI_PATH/txt/${lg}.train ***"

# split into train / valid / test
echo "*** Split into train / valid / test ***"
split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NTRAIN=$((NLINES - 10000));
    NVAL=$((NTRAIN + 5000));
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN             > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NVAL | tail -5000  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -5000                > $4;
}
split_data $WIKI_PATH/txt/$lg.all $WIKI_PATH/txt/$lg.train $WIKI_PATH/txt/$lg.valid $WIKI_PATH/txt/$lg.test


echo "*** Learn BPE from both wiki source (training set) and coco_dataset ***"
# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $NCODES $WIKI_PATH/txt/$lg.train $TRAIN_CAP > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"


# apply BPE codes
echo "Apply BPE codes..."
for split in train valid test; do
  $FASTBPE applybpe $PROCESSED_PATH/wiki/$lg.$split $WIKI_PATH/txt/$lg.$split $BPE_CODES
  $FASTBPE applybpe $PROCESSED_PATH/cap/cap.$split $CAP_PATH/raw_caption_$split.$lg $BPE_CODES
done


# extract vocab
echo "Extracting vocabulary from caption and wiki..."
$FASTBPE getvocab $PROCESSED_PATH/wiki/$lg.train $PROCESSED_PATH/cap/cap.train> $VOCAB


# binarize caption and wiki corpora
for split in train valid test; do
    python preprocess.py $VOCAB $PROCESSED_PATH/wiki/$lg.$split
    python preprocess.py $VOCAB $PROCESSED_PATH/cap/cap.$split
done