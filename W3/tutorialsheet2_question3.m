% import data sets used to predict quality of wine from chemical makeup
load('red_wine_quality_data.mat');
load('white_wine_quality_data.mat');

% 3a - shuffle data sets

% output of randperm function is to shuffle/reorder data sets
sd = 1;
rng(sd);
r = randperm(1599);

red_wine_x = red_wine_x(r,:);
red_wine_y = red_wine_y(r,:);
white_wine_x = white_wine_x(r,:);
white_wine_y = white_wine_y(r,:);

% calculated with nearest integer (75% train - 25% test)
red_wine_x_train = red_wine_x(1:1199, :);
red_wine_x_test = red_wine_x(1199:1599, :);

red_wine_y_train = red_wine_y(1:1199, :);
red_wine_y_test = red_wine_y(1199:1599, :);

white_wine_x_train = white_wine_x(1:1199, :);
white_wine_x_test = white_wine_x(1199:1599, :);

white_wine_y_train = white_wine_y(1:1199, :);
white_wine_y_test = white_wine_y(1199:1599, :);

red_wine_x_train_std = mapstd(red_wine_x_train.', 0,1).';
red_wine_y_train_std = mapstd(red_wine_y_train.', 0,1).';
red_wine_x_test_std = mapstd(red_wine_x_test.', 0,1).';
red_wine_y_test_std = mapstd(red_wine_y_test.', 0,1).';

white_wine_x_train_std = mapstd(white_wine_x_train.', 0,1).';
white_wine_y_train_std = mapstd(white_wine_y_train.', 0,1).';
white_wine_x_test_std = mapstd(white_wine_x_test.', 0,1).';
white_wine_y_test_std = mapstd(white_wine_y_test.', 0,1).';

% extract the first 5 standardized points
red_wine_x_test_std(1:5, :)
white_wine_x_test_std(1:5, :)

% 3b - fit a linear regression model to each training set

beta_red = mvregress(red_wine_x_train_std,red_wine_y_train_std);
beta_white = mvregress(white_wine_x_train_std,white_wine_y_train_std);

% the largest values in either direction have the greatest influence 
[~,x_red] = max(abs(beta_red));
[~,x_white] = max(abs(beta_white));
