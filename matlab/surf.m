function [featVector] = surf(I, SURFSize)

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
    % if anynan(obliq) == 1
    %     fprintf("%s\n", fname);
    % end

    % Curtose
    vetorAux = kurtosis(histograma, 0, 1);
    curt = vetorAux(1:size(vetorAux, 2));
    % if anynan(curt) == 1
    %      fprintf("%s\n", fname);
    % end   

    featVector = [vHist, media, desvPad, obliq, curt];
end