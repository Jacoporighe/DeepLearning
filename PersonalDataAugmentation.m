clear all
warning off

%scegli valore di datas in base a quale dataset vi serve
datas = 29;

%carica dataset
load(strcat('DatasColor_', int2str(datas)), 'DATA');
NF = size(DATA{3}, 1); %number of folds
DIV = DATA{3}; %divisione fra training e test set
DIM1 = DATA{4}; %numero di training pattern
DIM2 = DATA{5}; %numero di pattern
yE = DATA{2}; %label dei patterns
NX = DATA{1}; %immagini

%carica rete pre-trained
net = imagePretrainedNetwork("alexnet",Weights="none");
siz = [227 227];

%parametri rete neurale
miniBatchSize = 30;
learningRate = 1e-4;
metodoOptim = 'sgdm';
options = trainingOptions(metodoOptim, ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 30, ...
    'InitialLearnRate', learningRate, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
numIterationsPerEpoch = floor(DIM1 / miniBatchSize);

for fold = 1:NF
    close all force

    trainPattern = (DIV(fold, 1:DIM1));
    testPattern = (DIV(fold, DIM1+1:DIM2));
    y = yE(DIV(fold, 1:DIM1)); %training label
    yy = yE(DIV(fold, DIM1+1:DIM2)); %test label
    numClasses = max(y); %number of classes

    %creo il training set
    clear nome trainingImages
    for pattern = 1:DIM1
        IM = NX{DIV(fold, pattern)}; %singola data immagine

        IM = imresize(IM, [siz(1) siz(2)]); %resize immagini per rendere compatibili con CNN
        if size(IM, 3) == 1
            IM(:, :, 2) = IM;
            IM(:, :, 3) = IM(:, :, 1);
        end
        trainingImages{pattern} = IM;
    end

    % Apply custom augmentation
    [trainingImages, y] = augmentData(trainingImages, y);

    % Convert to 4D array for training
    trainingImages4D = cat(4, trainingImages{:});

    % Data augmentation for the training set
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...
        'RandXScale', [1 2], ...
        'RandYReflection', true, ...
        'RandYScale', [1 2], ...
        'RandRotation', [-10 10], ...
        'RandXTranslation', [0 5], ...
        'RandYTranslation', [0 5]);
    augmentedTrainingImages = augmentedImageSource(size(trainingImages4D, [1,2,3]), trainingImages4D, categorical(y), 'DataAugmentation', imageAugmenter);

    %tuning della rete
    layersTransfer = net.Layers(1:end-3);
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
        softmaxLayer
        classificationLayer];
    netTransfer = trainNetwork(augmentedTrainingImages, layers, options);

    %creo test set
    clear nome test testImages
    for pattern = ceil(DIM1)+1:ceil(DIM2)
        IM = NX{DIV(fold, pattern)}; %singola data immagine

        IM = imresize(IM, [siz(1) siz(2)]);
        if size(IM, 3) == 1
            IM(:, :, 2) = IM;
            IM(:, :, 3) = IM(:, :, 1);
        end
        testImages(:, :, :, pattern-ceil(DIM1)) = uint8(IM);
    end

    %classifico test patterns
    [outclass, score{fold}] = classify(netTransfer, testImages);

    %calcolo accuracy
    [a, b] = max(score{fold}');
    ACC(fold) = sum(b == yy) / length(yy);

    %salvate quello che vi serve
    %%%%%

end

function [augmentedImages, augmentedLabels] = augmentData(trainingImages, trainingLabels)
    % Augment data using various techniques
    imageSize = size(trainingImages{1});
    augmentedImages = {};
    augmentedLabels = [];
    
    for i = 1:length(trainingImages)
        img = trainingImages{i};
        label = trainingLabels(i);
        
        % Original image
        augmentedImages{end+1} = img;
        augmentedLabels(end+1) = label;
        
        % Horizontal Flip
        flippedImg = flip(img, 2);
        augmentedImages{end+1} = flippedImg;
        augmentedLabels(end+1) = label;
        
        % Random Rotation
        rotatedImg = imrotate(img, randi([-30, 30]), 'bilinear', 'crop');
        augmentedImages{end+1} = rotatedImg;
        augmentedLabels(end+1) = label;
        
        % Random Crop
        cropSize = imageSize(1:2) - randi([0, min(imageSize(1:2))-1], 1, 2);
        croppedImg = imcrop(img, [randi([0, imageSize(2) - cropSize(2)]), randi([0, imageSize(1) - cropSize(1)]), cropSize(2) - 1, cropSize(1) - 1]);
        croppedImg = imresize(croppedImg, imageSize(1:2));
        augmentedImages{end+1} = croppedImg;
        augmentedLabels(end+1) = label;
        
        % Shifting
        shiftedImg = imtranslate(img, [randi([-10, 10]), randi([-10, 10])]);
        augmentedImages{end+1} = shiftedImg;
        augmentedLabels(end+1) = label;
        
        % Color Jittering
        jitteredImg = jitterColorHSV(img, 'Contrast', 0.2, 'Saturation', 0.2, 'Brightness', 0.2);
        augmentedImages{end+1} = jitteredImg;
        augmentedLabels(end+1) = label;
        
        % Adding Noise
        noisyImg = imnoise(img, 'gaussian', 0, 0.01);
        augmentedImages{end+1} = noisyImg;
        augmentedLabels(end+1) = label;
        
        % PCA Jittering
        pcaJitteredImg = pcaJitter(img);
        augmentedImages{end+1} = pcaJitteredImg;
        augmentedLabels(end+1) = label;
    end
    
    augmentedImages = augmentedImages';
    augmentedLabels = augmentedLabels';
end

function jitteredImg = pcaJitter(img)
    % Perform PCA on the image and add jitter
    img = double(img);
    imgFlat = reshape(img, [], 3);
    [coeff, score, ~] = pca(imgFlat);
    jitter = randn(1, 3) * 0.1;
    jitteredImgFlat = score + jitter * coeff';
    jitteredImg = reshape(jitteredImgFlat, size(img));
    jitteredImg = uint8(jitteredImg);
end
