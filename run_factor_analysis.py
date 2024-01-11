import pandas

from factor_analyzer import FactorAnalyzer

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import matplotlib.pyplot as plt 
import numpy

dataset = pandas.read_csv("bfi_dataset.csv")

# print(dataset)

#test whether there is correlation in the data with bartlett and p-value
chi2 ,p=calculate_bartlett_sphericity(dataset)
# print(chi2, p)

#try 25 factors, get eigenvalues, which is the amount of variance the factor explains
machine = FactorAnalyzer(n_factors=25, rotation=None)
machine.fit(dataset)
ev, v = machine.get_eigenvalues()
print(ev)

plt.scatter(range(1,dataset.shape[1]+1),ev)
plt.plot(range(1,dataset.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

machine = FactorAnalyzer(n_factors=6, rotation=None)
machine.fit(dataset)
output = machine.loadings_
print(output)

machine = FactorAnalyzer(n_factors=5, rotation='varimax')
machine.fit(dataset)
factor_loadings = machine.loadings_
print(factor_loadings)

dataset = dataset.values

results = numpy.dot(dataset, output)

pandas.DataFrame(results).round().to_csv("results2.csv", index=False)