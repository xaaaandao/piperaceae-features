function save_features(filename, feature, label)    
    [file,msg]=fopen(filename, "a");
    if msg < 0
        error("error because: %s", msg);
    end
    for i=1:length(feature)
        fprintf(file, "%s ", num2str(feature(i)));
    end
    fprintf(file, "%s \n", label);
    fclose(file);
    fprintf("Saving ... %s\n", filename);
end
