function output = format_time(descriptor, output)
% FORMAT_TIME  Adiciona no diretório aonde serão salvos as features o
% horário e o descritor utilizado.
%   OUTPUT = format_time(descriptor, output) adiciona o horário e o
%   descritor que está sendo utilizado.
    dt_now = string(datetime);
    dt_now = strrep(dt_now, ":", "-");
    dt_now = strrep(dt_now, " ", "+");
    output = fullfile(output, dt_now, descriptor);
end

function label = get_label(foldername)
% GET_LABEL Captura a classe que pertence aquela amostra, baseado no nome
% da pasta daquela imagem.
%   LABEL = get_label(foldername) nome da pasta aonde será retirado a
%   classe.
    foldername = strsplit(foldername, "/");
    foldername = string(foldername(end));
    label = strrep(foldername, "f", "");
    label = str2num(label);
end

function extract_features(descriptor, input, output)
% EXTRACT_FEATURES Extrai as features das imagens presentes no diretório
% que foi passado por input. As features são salvas no diretório de output.
%   extract_features(descriptor, input, output) produz arquivos com as
%   características nas imagens.
    output = format_time(descriptor, output);

    % o segundo parametro indica qual tipo está sendo verificado
    if ~exist(output, "dir")
        mkdir(output);
    end
    
    % must be single quotes
    dirs = dir([input, '/**/*.jpeg']);
    labels = [];
    images = [];
    
    for i=1:height(dirs)
        label = dirs(i).folder;
        label = get_label(label);

        filename = fullfile(dirs(i).folder, dirs(i).name);
        image = imread(filename);

        switch descriptor
            case "lbp"
                features = lbp(image);
            case "surf"
                features = surf(image, 64);
            otherwise
                error("descriptor invalid");
        end
        % labels(end+1) = label;
        labels = [labels;string(label)];
        images = [images;filename];
    end
    save(descriptor, features, images, string(label), labels, output);
end

function save_dataset(descriptor, features, images, labels, output)
% SAVE_DATASET Salva as informações do dataset.
%   save_dataset(descriptor, features, images, label, labels, output) 
% salva informações do dataset.
    n_features = size(features);
    n_images = size(images);
    fold = max(labels);
    data = [descriptor, fold, string(n_images(1)), string(n_features(2)), string(1)];
    T = array2table(data, VariableNames={'descriptor','fold','labels','features','patches'});
    filename = fullfile(output, "dataset.csv");
    writetable(T, filename,"Delimiter",";","QuoteStrings","all");
end

function save(descriptor, features, images, label, labels, output)
% SAVE Salva as características extraídas e outras informações em
% arquivos CSV.
%   save(descriptor, features, images, label, labels, output) invoca
% as demais funções que realizam o salvamento das características e
% outras informações.
    save_samples(images, labels, output);
    save_features(descriptor, features, label, output);
    save_dataset(descriptor, features, images, labels, output);
end

function save_samples(images, labels, output)
% SAVE_SAMPLES Salva as amostras que tiveram suas características
% extraídas e as labels (classes ou folds) que essa imagens pertencem.
%   save_samples(images, labels, output) produz um arquivo CSV com
% as amostras utilizadas.
    T = table(images, labels);
    filename = fullfile(output, "samples.csv");
    writetable(T, filename,"Delimiter",";","QuoteStrings","all");
end

function save_features(descriptor, features, label, output)
% SAVE_FEATURES Salva as características extraídas em um arquivo .txt
%   save_features(descriptor, feature, label, output) produz um arquivo TXT com
% as características extraídas.
    filename = fullfile(output, string(strjoin([descriptor, ".txt"], "")));
    % a = append
    file = fopen(filename, "a");
    for i=1:length(features)
        fprintf(file, "%s ", num2str(features(i)));
    end
    fprintf(file, "%s \n", string(label));
    fclose(file);
end


function feature = lbp(image)
% LBP Extrai as características das imagens usando o Local Binary Pattern.
%   lbp(image) extrai as características de uma imagem usando LBP.
    lbpFeatures = extractLBPFeatures(image);
    numNeighbors = 8;
    numBins = numNeighbors*(numNeighbors-1)+3;
    lbpCellHists = reshape(lbpFeatures, numBins, []);
    feature = reshape(lbpCellHists, 1, []);
end


function [featVector] = surf(image, SURFSize)
% SURF Extrai as características das imagens usando o Speed Up Robust
% Features.
%   SURF(image, SURFSize) extrai as características de uma imagem usando SURF.    
    points = detectSURFFeatures( image );
    [histograma, valid_points] = extractFeatures(image, points, "SURFSize", SURFSize); 
    
                            
    % escreve QTDE. DESCRITORES na tela
    vHist =  size(histograma, 1);
    
    % media
    vetorAux = mean(histograma, 1);
    media =  vetorAux(1:size(vetorAux, 2));
    
    % desvio padrao
    vetorAux = std(histograma, 0, 1);
    desvPad =  vetorAux(1:size(vetorAux, 2));

    % Obliquidade
    vetorAux = skewness(histograma, 0, 1);
    obliq =  vetorAux(1:size(vetorAux, 2));
    if anynan(obliq) == 1
        fprintf("%s\n", filename);
    end

    % Curtose
    vetorAux = kurtosis(histograma, 0, 1);
    curt = vetorAux(1:size(vetorAux, 2));
    if anynan(curt) == 1
         fprintf("%s\n", filename);
    end   

    featVector = [vHist, media, desvPad, obliq, curt];
end


extract_features('lbp', './test/dataset/GRAYSCALE', 'a');
