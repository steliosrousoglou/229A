function [] = plotAll(X, Y, i1, i2, name1, name2, mu, sig)

cl0 = find(Y == 0);
cl1 = find(Y == 1);
cl2 = find(Y == 2);
cl3 = find(Y == 3);
cl4 = find(Y == 4);
cl5 = find(Y == 5);
cl6 = find(Y == 6);

plot(X(cl0, i1) + normrnd(mu, sig, size(X(cl0, i1))), X(cl0, i2) + normrnd(mu, sig, size(X(cl0, i2))), 'ko', 'MarkerFaceColor', 'w','MarkerSize', 7);
hold on;
plot(X(cl1, i1) + normrnd(mu, sig, size(X(cl1, i1))), X(cl1, i2) + normrnd(mu, sig, size(X(cl1, i2))), 'ko', 'MarkerFaceColor', 'g','MarkerSize', 7);
hold on;
plot(X(cl2, i1) + normrnd(mu, sig, size(X(cl2, i1))), X(cl2, i2) + normrnd(mu, sig, size(X(cl2, i2))), 'ko', 'MarkerFaceColor', 'b','MarkerSize', 7);
hold on;
plot(X(cl3, i1) + normrnd(mu, sig, size(X(cl3, i1))), X(cl3, i2) + normrnd(mu, sig, size(X(cl3, i2))), 'ko', 'MarkerFaceColor', 'k','MarkerSize', 7);
hold on;
plot(X(cl4, i1) + normrnd(mu, sig, size(X(cl4, i1))), X(cl4, i2) + normrnd(mu, sig, size(X(cl4, i2))), 'ko', 'MarkerFaceColor', 'm','MarkerSize', 7);
hold on;
plot(X(cl5, i1) + normrnd(mu, sig, size(X(cl5, i1))), X(cl5, i2) + normrnd(mu, sig, size(X(cl5, i2))), 'ko', 'MarkerFaceColor', 'c','MarkerSize', 7);
hold on;
plot(X(cl6, i1) + normrnd(mu, sig, size(X(cl6, i1))), X(cl6, i2) + normrnd(mu, sig, size(X(cl6, i2))), 'ko', 'MarkerFaceColor', 'r','MarkerSize', 7);
hold on;
xlabel(name1);
ylabel(name2);
legend('Never', '> 10 years ago', 'In last decade', 'In last year', 'In last month', 'In last week', 'In last day', 'Location','southeast');

end