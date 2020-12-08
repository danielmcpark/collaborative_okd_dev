

# Mincheol Park Frameworks
# Main training shell scripts

# Datasets:   cifar10, cifar100, imagenet
# Benchmarks: vgg, resnet
# Epochs: 200
# optims: SGD
# Batch_size: 64 (cifar10), 128 (cifar100), 256 (imagenet)
# weight_decay: 1e-4(L2)

DISKPATH='/mnt/disk3/'
IMAGENET=${DISKPATH}'imagenet/'
CINIC10=${DISKPATH}'cinic10/'
TINY_IMAGENET=${DISKPATH}'tiny-imagenet-200/'
CIFAR10=${DISKPATH}'cifar10'
CIFAR100=${DISKPATH}'cifar100'
CUB200=${DISKPATH}'cub200'
CARS196=${DISKPATH}'cars196'
STANFORD=${DISKPATH}'stanford'

LOGPATH=${DISKPATH}'logs/newkd/'
WRNPATH=${LOGPATH}'wideresnet/'
RESPATH=${LOGPATH}'resnet/'
DENSEPATH=${LOGPATH}'densenet/'
MOBILEPATH1=${LOGPATH}'mobilenetv1/'
MOBILEPATH2=${LOGPATH}'mobilenetv2/'


PROGRAM_NAME=`/usr/bin/basename "$0"`
echo shell arg 0: $0
echo USING BASENAME: ${PROGRAM_NAME}
arg_data=default
arg_arch=default

function print_usage(){
/bin/cat << EOF
Usage:
    ${PROGRAM_NAME} [-d arg_data] [-a arg_arch]
Option:
    -d, dataset
    -a, model
EOF
}
if [ $# -eq 0 ];
then
    print_usage
    exit 1
fi

while getopts "d:a:h" opt
do
    case $opt in
        d) arg_data=$OPTARG; echo "ARG DATA: $arg_data";;
        a) arg_arch=$OPTARG; echo "ARG ARCH: $arg_arch";;
        h) print_usage;;
    esac
done

# Cub200
if [ "$arg_data" = "cub200" ];
then
    if [ "$arg_arch" = "resnet18_img" ]
    then
        python3 metric_learning.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=1e-4 \
                --epochs=80 \
                --wd=4e-5 \
                --batch-size=64 \
                --test-batch-size=256 \
                --data=${CUB200} \
                --save=${RESPATH}'resnet18_original_metric_cub200' \
                --ngpu='cuda:0'
    fi
    if [ "$arg_arch" = "resnet50_img" ]
    then
        python3 metric_learning.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=1e-4 \
                --epochs=80 \
                --wd=4e-5 \
                --triplet_margin=0.2 \
                --batch-size=64 \
                --test-batch-size=256 \
                --data=${CUB200} \
                --save=${RESPATH}'resnet50_original_metric_em512_cub200' \
                --embedding_size=512 \
                --ngpu='cuda:0'
    fi
fi

if [ "$arg_data" = "cars196" ];
then
    if [ "$arg_arch" = "resnet18_img" ]
    then
        python3 metric_learning.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=1e-4 \
                --epochs=80 \
                --triplet_margin=0.2 \
                --wd=4e-5 \
                --batch-size=64 \
                --test-batch-size=256 \
                --data=${CARS196} \
                --save=${RESPATH}'resnet18_original_metric_cars196' \
                --ngpu='cuda:0'
    fi
    if [ "$arg_arch" = "resnet50_img" ]
    then
        python3 metric_learning.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=1e-4 \
                --epochs=80 \
                --triplet_margin=0.2 \
                --wd=4e-5 \
                --batch-size=64 \
                --test-batch-size=256 \
                --data=${CARS196} \
                --save=${RESPATH}'resnet50_original_metric_em512_cars196' \
                --ngpu='cuda:1'
    fi
fi

if [ "$arg_data" = "stanford" ];
then
    if [ "$arg_arch" = "resnet18_img" ]
    then
        python3 metric_learning.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=1e-4 \
                --epochs=80 \
                --wd=4e-5 \
                --batch-size=64 \
                --test-batch-size=256 \
                --data=${STANFORD} \
                --save=${RESPATH}'resnet18_original_metric_stanford' \
                --ngpu='cuda:0'
    fi
    if [ "$arg_arch" = "resnet50_img" ]
    then
        python3 metric_learning.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=1e-4 \
                --epochs=80 \
                --wd=4e-5 \
                --triplet_margin=0.2 \
                --batch-size=64 \
                --test-batch-size=256 \
                --data=${STANFORD} \
                --save=${RESPATH}'resnet50_original_metric_stanford' \
                --ngpu='cuda:0'
    fi
fi

