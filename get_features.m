function output = format_time(descriptor, output)
% FORMAT_TIME  Adiciona no diretório aonde serão salvos as features o
% horário e o descritor utilizado.
%   OUTPUT = format_time(descriptor, output) adiciona o horário e o
%   descritor que está sendo utilizado.
    dt_now = string(datetime);
    dt_now = strrep(dt_now, ":", "-");
    dt_now = strrep(dt_now, " ", "+");
    output = fullfile(output, descriptor);
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

function extract_features(descriptor, input, minimum, name, output)
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
    features = [];
    
    for i=1:height(dirs)
        label = dirs(i).folder;
        label = [get_label(label)];

        filename = fullfile(dirs(i).folder, dirs(i).name);
        image = imread(filename);
        % 
        % [h, w] = size(image);
        w = size(image, 1);
        h = size(image, 2);

        image = rgb2gray(image); % Convert to grayscale
        % imadjust(image); % Adjust contrast


        % fprintf("%s\n", filename);
        switch descriptor
            case "lbp"
                feature = lbp(image);
            case "surf"
                feature = surf(image, 128);
            otherwise
                error("descriptor invalid");
        end
        feature(end+1) = label;
        features = [features; feature]; % Concatenate rows for each iteration
        labels = [labels; label];
        images = [images; string(filename)];
    end
    save(descriptor, features, h, images, labels, minimum, name, output, w);
end

function save_dataset(descriptor, features, height, images, labels, ...
    minimum, name, output, width)
    % SAVE_DATASET Salva as informações do dataset.
    n_features = size(features, 2); % Número de colunas de features
    n_images = numel(images); % Número total de imagens
    fold = max(labels); % Número máximo de labels (supõe classes sequenciais)
    patches = 1; % Valor padrão (ajustável)

    % model = descritpor
   T = table({'GRAYSCALE'}, 1.2, fold, {'txt'}, height, patches, n_features, n_images, ...
              {descriptor}, {name}, minimum, width, n_images, ...
              'VariableNames', {'color', 'contrast', 'fold', 'format', 'height', ...
                                'patch', 'count_features', 'count_samples', 'model', ...
                                'name', 'minimum', 'width', 'count_samples_patch'});

    % Salva a tabela como arquivo CSV
    filename = fullfile(output, "dataset.csv");
    writetable(T, filename, "Delimiter", ";", "QuoteStrings", true);
end


function save(descriptor, features, h, images, labels, minimum, name, output, w)
% SAVE Salva as características extraídas e outras informações em
% arquivos CSV.
%   save(descriptor, features, images, label, labels, output) invoca
% as demais funções que realizam o salvamento das características e
% outras informações.
    save_samples(images, labels, output);
    save_features(descriptor, features, output);
    save_dataset(descriptor, features, h, images, labels, ...
    minimum, name, output, w);
end

function save_samples(filename, labels, output)
% SAVE_SAMPLES Salva as amostras que tiveram suas características
% extraídas e as labels (classes ou folds) que essa imagens pertencem.
%   save_samples(images, labels, output) produz um arquivo CSV com
% as amostras utilizadas.
    T = table(filename, labels, labels,   'VariableNames', {'filename', 'fold', 'specific_epithet'});
    fname = fullfile(output, "samples.csv");
    writetable(T, fname,"Delimiter",";","QuoteStrings","all");
end

function save_features(descriptor, features, output)
% SAVE_FEATURES Salva as características extraídas em um arquivo .txt
%   save_features(descriptor, feature, label, output) produz um arquivo TXT com
% as características extraídas.
    filename = fullfile(output, string(strjoin([descriptor, ".txt"], "")));
    % a = append
    file = fopen(filename, "w");
    
    for i = 1:length(features)
        fprintf(file, "%s\n", num2str(features(i,:)));
    end
    fprintf("file: %s\n", filename);
    
    fclose(file);
end


function [feature] = lbp(image)
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
    histograma(isnan(histograma)) = 0;
                            
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
    if isnan(obliq) == 1
        fprintf("%d\n", points.Count);
        fprintf("ruim\n");
        % pause;
    end

    % Curtose
    vetorAux = kurtosis(histograma, 0, 1);
    curt = vetorAux(1:size(vetorAux, 2));
    if isnan(curt) == 1
        fprintf("ruim\n");
        % pause;
    end

    featVector = [vHist, media, desvPad, obliq, curt];
end

extract_features('surf', '/home/xandao/Documentos/pr_dataset+5/RGB/256/original', 5, 'pr_dataset', '/home/xandao/Documentos/pr_dataset+5/GRAYSCALE/256');
extract_features('surf', '/home/xandao/Documentos/pr_dataset+5/RGB/400/original', 5, 'pr_dataset', '/home/xandao/Documentos/pr_dataset+5/GRAYSCALE/400');
extract_features('surf', '/home/xandao/Documentos/pr_dataset+5/RGB/512/original', 5, 'pr_dataset', '/home/xandao/Documentos/pr_dataset+5/GRAYSCALE/512');
extract_features('surf', '/home/xandao/Documentos/pr_dataset+10/RGB/256/original', 10, 'pr_dataset', '/home/xandao/Documentos/pr_dataset+10/GRAYSCALE/256');
extract_features('surf', '/home/xandao/Documentos/pr_dataset+10/RGB/400/original', 10, 'pr_dataset', '/home/xandao/Documentos/pr_dataset+10/GRAYSCALE/400');
extract_features('surf', '/home/xandao/Documentos/pr_dataset+10/RGB/512/original', 10, 'pr_dataset', '/home/xandao/Documentos/pr_dataset+10/GRAYSCALE/512');
extract_features('surf', '/home/xandao/Documentos/pr_dataset+20/RGB/256/original', 20, 'pr_dataset', '/home/xandao/Documentos/pr_dataset+20/GRAYSCALE/256');
extract_features('surf', '/home/xandao/Documentos/pr_dataset+20/RGB/400/original', 20, 'pr_dataset', '/home/xandao/Documentos/pr_dataset+20/GRAYSCALE/400');
extract_features('surf', '/home/xandao/Documentos/pr_dataset+20/RGB/512/original/', 20, 'pr_dataset', '/home/xandao/Documentos/pr_dataset+20/GRAYSCALE/512'); 