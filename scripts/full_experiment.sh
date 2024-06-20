DATASET="multiturn_chat_0.8m-chinese-zhtw"
# DATASET="zhtw-sentence-error-correction"
# DATASET="generated_chat_0.4m-chinese-zhtw"
# DATASET="train_1m-chinese-zhtw"
# DATASET="dolly-15k-chinese-zhtw"

EPOCHS=4
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=32
LEARNING_RATE=2e-5
NUM_OF_IMG=10000
SUBMIT=false

for i in "$@"
do
case $i in
    --submit=*)
    SUBMIT="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

source .data_prep/bin/activate
python extract_sentences.py $DATASET
python generate_custom_dataset.py $NUM_OF_IMG
deactivate

source .tinyllava_factory/bin/activate
cd TinyLLaVA_Factory
bash scripts/train/custom_finetune.sh $EPOCHS $BATCH_SIZE $GRADIENT_ACCUMULATION_STEPS $LEARNING_RATE
cd ..

python inference.py

if [ "$SUBMIT" = true ] ; then
    SUBMIT_MSG=$(printf "dataset: $DATASET\nepoch: $EPOCHS\nbatch size: $BATCH_SIZE\ngradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS\nlearning rate: $LEARNING_RATE")
    kaggle competitions submit zh-tw-tv-show-caption-recognition-using-lm-ms -f submission.csv -m "$SUBMIT_MSG"
fi