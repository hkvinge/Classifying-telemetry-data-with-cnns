% Performs a continuous wavelet convolution on time series

% Put limit on number of pictures to produce
max_train = 5000;
max_test = 1000;
fourier_clip = 20;

% Load time series data
temps_train = csvread("temps_train.csv");
temps_test = csvread("temps_test.csv");
labels_train = csvread("label_train.csv");
labels_test = csvread("label_test.csv");

% Plot a few time series
x = 1:size(temps_train,1);
figure();
plot(x,temps_train(:,1));
set(gca,'FontSize',17)
figure()
plot(x,temps_train(:,2));
set(gca,'FontSize',17)
labels_train(1)
labels_train(2)

pictures_train = zeros(max_train*37,size(temps_train,1));
labels_train_revised = zeros(max_train,1);

pictures_temp_train = zeros(37,size(temps_train,1),max_train);
fourier_coefficients_train = zeros(2*fourier_clip,max_train);

for i=0:max_train-1
    ts = abs(cwt(temps_train(:,i+1)));
    pictures_temp_train(:,:,i+1) = ts;  
    f_real = real(fft(temps_train(:,i+1)));
    f_image = imag(fft(temps_train(:,i+1)));
    f_real = f_real(1:fourier_clip);
    fourier_coefficients_train(1:fourier_clip,i+1) = f_real;
    f_image = f_image(1:fourier_clip);
    fourier_coefficients_train(fourier_clip+1:2*fourier_clip,i+1) = f_image;
end


% Take mean of rows
fourier_coefficients_train = fourier_coefficients_train' - mean(fourier_coefficients_train');
fourier_coefficients_train = fourier_coefficients_train';

figure();
x = 1:2*fourier_clip;
plot(x,fourier_coefficients_train(:,1));
set(gca,'FontSize',17)
figure();
plot(x,fourier_coefficients_train(:,2));
set(gca,'FontSize',17)

for i=1:37
    for j=1:96
        pictures_temp_train(i,j,:) = pictures_temp_train(i,j,:) - mean(pictures_temp_train(i,j,:))*ones(size(pictures_temp_train(i,j,:)));
        Var = var(pictures_temp_train(i,j,:));
        pictures_temp_train(i,j,:) = (1/Var)*pictures_temp_train(i,j,:);
    end
end

for i=0:max_train-1
    if (mod(i,100) == 0)
        i
    end
    pictures_train(i*37 + 1:(i+1)*37,:) = pictures_temp_train(:,:,i+1);
    if (labels_train(i+1) > 0)
        labels_train_revised(i+1) = 1;
    else
        labels_train_revised(i+1) = 0;
    end
end

csvwrite("temps_pictures_train.csv",pictures_train);
csvwrite("labels_train_revised.csv",labels_train_revised);
csvwrite("fourier_coefficients_train.csv",fourier_coefficients_train);

pictures_test = zeros(max_test*37,size(temps_test,1));
labels_test_revised = zeros(max_test,1);

pictures_temp_test = zeros(37,size(temps_test,1),max_test);
fourier_coefficients_test = zeros(2*fourier_clip,max_test);

for i=0:max_test-1
    pictures_temp_test(:,:,i+1) = abs(cwt(temps_test(:,i+1)));
    f_real = real(fft(temps_test(:,i+1)));
    f_image = imag(fft(temps_test(:,i+1)));
    f_real = f_real(1:fourier_clip);
    fourier_coefficients_test(1:fourier_clip,i+1) = f_real;
    f_image = f_image(1:fourier_clip);
    fourier_coefficients_test(fourier_clip+1:2*fourier_clip,i+1) = f_image;
end

% Take mean of rows
fourier_coefficients_test = fourier_coefficients_test' - mean(fourier_coefficients_test');
fourier_coefficients_test = fourier_coefficients_test';

for i=1:37
    for j=1:96
        pictures_temp_test(i,j,:) = pictures_temp_test(i,j,:) - mean(pictures_temp_test(i,j,:))*ones(size(pictures_temp_test(i,j,:)));
        Var = var(pictures_temp_test(i,j,:))
        pictures_temp_test(i,j,:) = (1/Var)*pictures_temp_test(i,j,:);
    end
end

for i=0:max_test-1
    if (mod(i,100) == 0)
        i
    end
    pictures_test(i*37 + 1:(i+1)*37,:) = pictures_temp_test(:,:,i+1);
    if (labels_test(i+1) > 0)
        labels_test_revised(i+1) = 1;
    else
        labels_test_revised(i+1) = 0;
    end
end

figure();
imagesc(pictures_train(1:37,:));
figure();
imagesc(pictures_train(38:74,:));
figure();
imagesc(pictures_test(1:37,:));
figure();
imagesc(pictures_test(38:74,:));

% Mean center data

csvwrite("temps_pictures_test.csv",pictures_test);
csvwrite("labels_test_revised.csv",labels_test_revised);
csvwrite("fourier_coefficients_test.csv",fourier_coefficients_test);

    