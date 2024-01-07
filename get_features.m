function get_features()
    path_base = "/home/xandao/Imagens";
    colors = ["GRAYSCALE"];
    datasets = ["br_dataset"];
    image_sizes = ["256", "400"];
    levels = ["specific_epithet_trusted"];
    minimum_images = ["5"];
    for dataset=datasets
        for color=colors
            for level=levels
                for image_size=image_sizes
                    for minimum_image=minimum_images
                        a(color, dataset, image_size, level, minimum_image, path_base);
                    end
                end
            end            
        end
    end
end

function a(color_mode, d, image_size, l, min_image, path_base)
    if d=="regions_dataset"
        regions = ["Norte", "Nordeste", "Centro-Oeste", "Sul", "Sudeste"];
        for r=regions
            path_in = fullfile(path_base, d, color_mode, l, r, image_size, min_image);
            path_out = fullfile(path_base, strcat(d, "_features"), color_mode, l, r, image_size, min_image);
            if ~exist(path_out, 'dir')
                mkdir(path_out)
            end
            delete_files(path_out);
            [extractor, n_features, total_samples] = extract_features(path_in, path_out);
            region=[r;r;r];
            [color, dataset, height, width, level, minimum_image, input_path, output_path] = informations(color_mode, d, image_size, image_size, l, min_image, path_in, path_out);
            T = table(extractor, n_features, total_samples, color, height, width, level, minimum_image, input_path, output_path, region, dataset);
            save_table_info(path_out, T);
        end
    else
        path_in = fullfile(path_base, d, color_mode, l, image_size, min_image);
        path_out = fullfile(path_base, strcat(d, "_features"), color_mode, image_size, l, min_image);
        if ~exist(path_out, 'dir')
            mkdir(path_out)
        end
        delete_files(path_out);
        [extractor, n_features, total_samples] = extract_features(path_in, path_out);
%         [color, dataset, height, width, level, minimum_image, input_path, output_path] = informations(color_mode, d, image_size, image_size, l, min_image, path_in, path_out);
%         T = table(extractor, n_features, total_samples, color, height, width, level, minimum_image, input_path, output_path, dataset);
%         save_table_info(path_out, T);
    end
end


function save_table_info(path_out, T)
    filename = fullfile(path_out, 'info.csv');
    fprintf("%s created\n", string(filename));
    writetable(T, filename);
end


function [color, dataset, height, width, level, minimum_image, input_path, output_path] = informations(c, d, h, w, l, m, in, out)
    color = [c;c;c];
    dataset = [d;d;d];
    height = [h;h;h];
    width = [w;w;w];
    level = [l;l;l];
    minimum_image = [m;m;m];
    input_path = [in;in;in];
    output_path = [out;out;out];
end


function [extractor, n_features, total_samples] = extract_features(path_in, path_out)
    labels=[];
    inputs=[];
    list_dir = dir(path_in);
    folders = list_dir([list_dir.isdir]); % A structure with extra info.        
    folders_name = folders(3:end);
    total_samples = 0;
    for i=1:height(folders_name)
        path_folder = fullfile(folders_name(i).folder, folders_name(i).name);
        list_images = dir(path_folder);
        for j=1:height(list_images)
            path_to_image =  fullfile(list_images(j).folder, list_images(j).name);
            if contains(path_to_image, "jpeg", "IgnoreCase", true)
%                 fprintf("path: %s j: %d total: %d\n", path_folder, path_to_image, height(list_images));
                img = imread(path_to_image);
%                 img = histeq(img, [0 1]);
%                 img = imbinarize(img);
                labels = [labels;string(i)];
                inputs = [inputs;string(path_to_image)];                

%                 feature = lbp(img);
%                 filename_lbp = fullfile(path_out, "lbp.txt");
%                 fileout(filename_lbp, feature, string(i));
%                 n_features_lbp = string(length(feature));
                
                feature = surf(img, path_to_image, 64);
                filename_surf = fullfile(path_out, "surf64.txt");
                fileout(filename_surf, feature, string(i));
                n_features_surf64 = string(length(feature));

%                 feature = surf(img, path_to_image, 128);
%                 filename_surf = fullfile(path_out, "surf128.txt");
%                 fileout(filename_surf, feature, string(i));
                n_features_surf128 = string(length(feature));

                total_samples = total_samples + 1;
            end
        end
    end 
    extractor=["lbp"; "surf64"; "surf128"];
    n_features=[n_features_lbp; n_features_surf64; n_features_surf128];
    total_samples=[string(total_samples); string(total_samples); string(total_samples)];
% 
%     T = table(labels, inputs);
%     filename = fullfile(path_out, 'info_samples.csv');
%     fprintf("%s created\n", string(filename));
%     writetable(T, filename);
end


function delete_files(path)
    for fname=["surf.txt", "surf128.txt", "lbp.txt"]
        fprintf("delete file %s\n", string(fullfile(path, fname)));
        delete(fullfile(path, fname));
    end
end


function fileout(filename, feature, label)    
    file = fopen(filename, "a");
    for i=1:length(feature)
        fprintf(file, "%s ", num2str(feature(i)));
    end
    fprintf(file, "%s \n", label);
    fclose(file);
end


function feature = lbp(image)
    lbpFeatures = extractLBPFeatures(image);
    numNeighbors = 8;
    numBins = numNeighbors*(numNeighbors-1)+3;
    lbpCellHists = reshape(lbpFeatures, numBins, []);
    feature = reshape(lbpCellHists, 1, []);
end


function [featVector] = surf(I, fname, SURFSize)

    %I = single( I );		

    points = detectSURFFeatures( I );
    [histograma, valid_points] = extractFeatures(I, points, 'SURFSize', SURFSize); 
    
                            
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
        fprintf("%s\n", fname);
    end

    % Curtose
    vetorAux = kurtosis(histograma, 0, 1);
    curt = vetorAux(1:size(vetorAux, 2));
    if anynan(curt) == 1
         fprintf("%s\n", fname);
    end   

    featVector = [vHist, media, desvPad, obliq, curt];
end
