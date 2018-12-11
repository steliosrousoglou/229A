# true positives / (true positives + false positives)
function [ans] = precision(y, predictions, p)
    t = find(y == p);
    a = find(predictions' == p);
    if length(t) == 0
        ans = NaN;
    else
        ans = length(intersect(t, a))/length(union(t, a));
    endif
end