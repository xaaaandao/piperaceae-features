function get_features()
    contrast=0;
    % use double asterik **/*.jpeg tto find files recursively
    path='/home/xandao/mygit/pr_dataset/GRAYSCALE/specific_epithet_trusted/256/5/**/*.jpeg';
    output='/home/xandao/pr_dataset_contraste/GRAYSCALE/specific_epithet_trusted/256/5/';
    delete_features(output);
    extract_features(contrast, path, output);

    
end



function extract_features(contrast, path, output)
    files=dir(path);
    % create csv files to store informations
    info_dataset_lbp=create_info_csv();
    info_dataset_surf64=create_info_csv();
    info_dataset_surf128=create_info_csv();
    info_levels=create_info_levels();
    info_samples=create_info_samples(length(files));
    total_samples=0;
    for k=1:length(files)
        % get filename
        fn=fullfile(files(k).folder, files(k).name);

        % extract f to use as label
        pat=digitsPattern;
        f=extract(files(k).folder,pat);
        label=f(end);
        label=label{1};

        fprintf("%d-%d filename: %s\n", k, length(files), files(k).name);

        % load image
        img=imread(fn,'jpeg');


        if ~exist(output, "dir")
            mkdir(output);
        end 
        
        if contrast > 0
            img = imcontrast(img);
            path=fullfile(output, "images", "matlab", files(k).folder, files(k).name);
            imsave(img, path);
        end

        for p=["lbp", "surf64", "surf128"]
            path_features=fullfile(output, "features", strcat(p,"+matlab"));
            if ~exist(path_features, "dir")
                mkdir(path_features);
            end 
        end
        
        total_samples=total_samples+1;
        features = lbp(img);    
        fname=fullfile(output, "features", "lbp+matlab", "lbp.txt");
        save_features(fname, features, label);
        info_dataset_lbp(1,:) = {width(features), total_samples, contrast, "lbp", "grayscale", height(img), width(img), 1};

        features = surf(img, 64);
        fname=fullfile(output, "features", "surf64+matlab", "surf64.txt");
        save_features(fname, features, label);
        info_dataset_surf64(1,:) = {width(features), total_samples, contrast, "surf64", "grayscale", height(img), width(img), 1};

        features = surf(img, 128);
        fname=fullfile(output, "features", "surf128+matlab", "surf128.txt");
        save_features(fname, features, label);
        info_dataset_surf128(1,:) = {width(features), total_samples, contrast, "surf128", "grayscale", height(img), width(img), 1};

        info_samples(k,:)={string(fn), str2double(label)};
        info_levels(k,:)={label, files(k).folder, 1};
    end
    % save alls csv 
    save_csv(info_dataset_lbp, "lbp", "info", output, true);
    save_csv(info_dataset_surf64, "surf64", "info", output, true);
    save_csv(info_dataset_surf128, "surf128", "info", output, true);
    info_levels = count_sample_by_level(info_levels);
    for d=["lbp", "surf64", "surf128"]
        save_csv(info_levels, d, "info_levels", output, false);
        save_csv(info_samples, d, "info_samples", output, false);
    end
end

% counts the number of samples of the same class.
function [info_levels] = count_sample_by_level(info_levels)
    info_levels = groupcounts(info_levels, ["f","levels"]);
    info_levels = renamevars(info_levels, "GroupCount", "count");
    info_levels = renamevars(info_levels, "Percent", "percent");
end

% save table in csv format
function save_csv(csv, descriptor, filename, output, transpose)
    fname=fullfile(output, "features", strcat(descriptor, "+matlab"), strcat(filename, ".csv"));
    if exist(fname, "file")
        delete(fname);
    end

    fprintf("Saving.... %s\n", string(fname));
    if transpose==true
        data=rows2vars(csv);
        writetable(data, fname, 'Delimiter', ';', 'QuoteStrings', 'all', 'WriteVariableNames', false);
    else
        data=csv;
        writetable(data, fname, 'Delimiter', ';', 'QuoteStrings', 'all', 'WriteVariableNames', true);
    end
end

% delete features file if exists
function delete_features(path)
    % surf64, surf128, lbp must be between double quotes
    for fname=["surf64", "surf128", "lbp"]
        fname=fullfile(path, "features", strcat(fname, "+matlab"), strcat(fname, ".txt"));
        fprintf("delete file %s\n", string(fname));
        if exist(fname, "file")
            delete(fname);
        end
    end
end

% create table info csv
function [info_dataset] = create_info_csv()
    sz=[1 8];
    var_types=["int16", "int16", "double", "string", "string", "int16", "int16", "int16"];
    var_names=["n_features", "total_samples", "contrast", "descriptor", "color", "height", "width", "n_patches"];
    info_dataset=table('Size', sz,'VariableTypes',var_types, 'VariableNames',var_names);
end

% create table levels
function [info_levels] = create_info_levels()
    f={};
    levels={};
    count=[];
    info_levels=table(f, levels, count);
end

% create table samples
function [info_samples] = create_info_samples(count_total)
    sz=[count_total 2];
    var_types=["string", "string"];
    var_names=["filename", "label"];
    info_samples=table('Size', sz,'VariableTypes',var_types, 'VariableNames',var_names);
end

