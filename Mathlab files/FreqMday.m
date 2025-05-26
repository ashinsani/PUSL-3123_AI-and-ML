clear;
clc;
close all;

% Parameters
num_users = 10;  % No. of users
trainSplit = 0.6; % Training split (60-40)
imposterSplit = 25; % Imposter samples ratio

% Load data
data = cell(1, num_users); % Cell array to store user data
for i = 1:num_users
    filename = fullfile('Data', sprintf('U%02d_Acc_FreqD_MDay.mat', i)); % File path
    user_data = load(filename);  % Load the .mat file
    data{i} = user_data.Acc_FD_Feat_Vec;  % Correct variable name based on your data
end

% Training and evaluation for each user
for user_id = 1:num_users
    user_data = data{user_id};
    numUserSamples = size(user_data, 1);
    numUsetrainSamples = round(trainSplit * numUserSamples);

    % Split user data into train and test sets
    X_train = user_data(1:numUsetrainSamples, :);
    X_test = user_data(numUsetrainSamples + 1:end, :);
    Y_train = ones(1, size(X_train, 1));  % Label for genuine user data (1)
    Y_test = ones(1, size(X_test, 1));   % Label for genuine user data (1)

    % Add imposter data to train and test sets
    for imposter_id = 1:num_users
        if imposter_id ~= user_id
            imposter_data = data{imposter_id};
            X_train = [X_train; imposter_data(1:imposterSplit, :)];  % Add imposter samples to train data
            X_test = [X_test; imposter_data(imposterSplit + 1:end, :)];  % Add imposter samples to test data
            Y_train = [Y_train, zeros(1, imposterSplit)];  % Label imposter data as 0
            Y_test = [Y_test, zeros(1, size(imposter_data(imposterSplit + 1:end, :), 1))]; % Label imposter data as 0
        end
    end

    % Shuffle training data
    trainIdx = randperm(size(X_train, 1));
    X_train = X_train(trainIdx, :);
    Y_train = Y_train(trainIdx);

    % Initialize metrics storage
    total_accuracy = [];
    total_precision = [];
    total_recall = [];
    total_f1score = [];
    total_FAR = [];
    total_FRR = [];

    % Training and evaluation for each run (you can increase num_runs for multiple evaluations)
    for num_runs = 1:1
        hiddenLayerSize = [15];% Number of neurons in the hidden layer
        net = feedforwardnet(hiddenLayerSize);
        net.trainParam.Ir = 0.02;
        net.trainParam.epochs = 75;
        net.performParam.regularization = 0.2;
        net = train(net, X_train', Y_train);  % Train the network

        % Evaluate on test data
        Y_pred = net(X_test');
        Y_pred_class = Y_pred > 0.5;  % Convert outputs to binary class labels

        % Metrics calculation
        accuracy = mean(Y_test == Y_pred_class) * 100;
        TP = sum((Y_test == 1) & (Y_pred_class == 1));
        FP = sum((Y_test == 0) & (Y_pred_class == 1));
        FN = sum((Y_test == 1) & (Y_pred_class == 0));
        TN = sum((Y_test == 0) & (Y_pred_class == 0));

        precision = TP / (TP + FP + eps);
        recall = TP / (TP + FN + eps);
        F1 = 2 * (precision * recall) / (precision + recall + eps);
        FAR = FP / (FP + TN + eps);
        FRR = FN / (FN + TP + eps);

        % Store results
        total_accuracy = [total_accuracy, accuracy];
        total_precision = [total_precision, precision];
        total_recall = [total_recall, recall];
        total_f1score = [total_f1score, F1];
        total_FAR = [total_FAR, FAR];
        total_FRR = [total_FRR, FRR];
    end

    % Display averaged results
    fprintf('User %d Results: \n', user_id);
    fprintf('Average Accuracy: %.2f%%\n', mean(total_accuracy));
    fprintf('Average Precision: %.2f\n', mean(total_precision));
    fprintf('Average Recall: %.2f\n', mean(total_recall));
    fprintf('Average F1-score: %.2f\n', mean(total_f1score));
    fprintf('Average FAR: %.2f%%\n', mean(total_FAR) * 100);
    fprintf('Average FRR: %.2f%%\n', mean(total_FRR) * 100);
    disp('--------------------------------');
end
