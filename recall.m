# true positives / (true positives + false negatives)
function [ans] = recall(y, predictions, p)
    t = find(y == p);
    a = find(predictions'== p);
    if length(t) == 0
        ans = NaN;
    else
        ans = length(intersect(t, a))/ (length(intersect(find(predictions'!= p), t)) + length(intersect(t, a)));
    endif
end