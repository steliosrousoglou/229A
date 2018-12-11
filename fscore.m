function [f1] = fscore(y, predictions, p)
    pr = precision(y, predictions, p);
    re = recall(y, predictions, p);
    if pr == 0 || re == 0
        f1 = NaN;
    else
        f1 = 2 * (pr * re) / (pr + re);
    endif
    
end