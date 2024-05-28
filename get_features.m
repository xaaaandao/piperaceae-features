function [output] = format_time(descriptor, output)
% FORMAT_TIME  Adiciona no diretório aonde serão salvos as features o
% horário e o descritor utilizado.
%   OUTPUT = format_time(descriptor, output) adiciona o horário e o
%   descritor que está sendo utilizado.
    dt_now = string(datetime);
    dt_now = strrep(dt_now, ':', '-');
    dt_now = strrep(dt_now, ' ', '+');
    [output] = fullfile(output, dt_now, descriptor);
end

function [label] = get_label(foldername)
% GET_LABEL Captura a classe que pertence aquela amostra, baseado no nome
% da pasta daquela imagem.
%   LABEL = get_label(foldername) nome da pasta aonde será retirado a
%   classe.
    foldername = strsplit(foldername, '/');
    foldername = string(foldername(end));
    label = strrep(foldername, 'f', '');
    [label] = str2num(label);
end

function extract_features(descriptor, input, output)
% EXTRACT_FEATURES Extrai as features das imagens presentes no diretório
% que foi passado por input. As features são salvas no diretório de output.
%   extract_features(descriptor, input, output) produz arquivos com as
%   características nas imagens.
    output = format_time(descriptor, output);

    % o segundo parametro indica qual tipo está sendo verificado
    if ~exist(output, 'dir')
        mkdir(output);
    end
    
    dirs = dir([input, '/**/*.jpeg']);
    display(dirs); 
    
    for i=1:height(dirs)
        display(fullfile(dirs(i).folder, dirs(i).name));
        label = dirs(i).folder;
        label = get_label(label);

        filename = fullfile(dirs(i).folder, dirs(i).name);
        image = imread(filename);

        switch descriptor
            case 'lbp'
                features = lbp(image);
            case 'surf'
                features = surf(image, 64);
            otherwise
                error('descriptor invalid');
        end

        save(descriptor, features, string(label), output);
    end
end


function save(descriptor, feature, label, output)
% SAVE Salva as características extraídas em um arquivo .txt
%   save(descriptor, feature, label, output) produz um arquivo TXT com
% as características extraídas.
    filename = fullfile(output, string(strjoin([descriptor, ".txt"], '')));
    % a = append
    file = fopen(filename, "a");
    for i=1:length(feature)
        fprintf(file, "%s ", num2str(feature(i)));
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
    [histograma, valid_points] = extractFeatures(image, points, 'SURFSize', SURFSize); 
    
                            
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
