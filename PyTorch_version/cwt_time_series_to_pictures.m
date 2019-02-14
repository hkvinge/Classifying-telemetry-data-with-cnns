% Performs a continuous wavelet convolution on time series

% Put limit on number of pictures to produce
max_train = 4000;
max_test = 1000;

% Load time series data
temps_train = csvread("temps_train.csv");
temps_test = csvread("temps_test.csv");
labels_train = csvread("label_train.csv");
labels_test = csvread("label_test.csv");

% Plot a few time series
x = 1:size(temps_train,1);
figure();
plot(x,temps_train(:,1),x,temps_train(:,100),x,temps_train(:,200));

pictures_train = zeros(max_train*27,size(temps_train,1));
labels_train_revised = zeros(max_train,1);

pictures_temp_train = zeros(27,size(temps_train,1),max_train);

for i=0:max_train-1
    pictures_temp_train(:,:,i+1) = abs(cwt(temps_train(:,i+1)));
end

for i=1:27
    for j=1:48
        pictures_temp_train(i,j,:) = pictures_temp_train(i,j,:) - mean(pictures_temp_train(i,j,:))*ones(size(pictures_temp_train(i,j,:)));
        Var = var(pictures_temp_train(i,j,:))
        pictures_temp_train(i,j,:) = (1/Var)*pictures_temp_train(i,j,:);
    end
end

for i=0:max_train-1
    if (mod(i,100) == 0)
        i
    end
    pictures_train(i*27 + 1:(i+1)*27,:) = pictures_temp_train(:,:,i+1);
    if (labels_train(i+1) > 0)
        labels_train_revised(i+1) = 1;
    else
        labels_train_revised(i+1) = 0;
    end
end

csvwrite("temps_pictures_train.csv",pictures_train);
csvwrite("labels_train_revised.csv",labels_train_revised);

pictures_test = zeros(max_test*27,size(temps_test,1));
labels_test_revised = zeros(max_test,1);

pictures_temp_test = zeros(27,size(temps_test,1),max_test);

for i=0:max_test-1
    pictures_temp_test(:,:,i+1) = abs(cwt(temps_test(:,i+1)));
end

for i=1:27
    for j=1:48
        pictures_temp_test(i,j,:) = pictures_temp_test(i,j,:) - mean(pictures_temp_test(i,j,:))*ones(size(pictures_temp_test(i,j,:)));
        Var = var(pictures_temp_test(i,j,:))
        pictures_temp_test(i,j,:) = (1/Var)*pictures_temp_test(i,j,:);
    end
end

for i=0:max_test-1
    if (mod(i,100) == 0)
        i
    end
    pictures_test(i*27 + 1:(i+1)*27,:) = pictures_temp_test(:,:,i+1);
    if (labels_test(i+1) > 0)
        labels_test_revised(i+1) = 1;
    else
        labels_test_revised(i+1) = 0;
    end
end

figure();
imagesc(pictures_train(1:27,:));
figure();
imagesc(pictures_train(28:54,:));
figure();
imagesc(pictures_test(1:27,:));
figure();
imagesc(pictures_test(28:54,:));

% Mean center data

csvwrite("temps_pictures_test.csv",pictures_test);
csvwrite("labels_test_revised.csv",labels_test_revised);

    