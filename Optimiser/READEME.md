NOTE: The CDFTSVM contains a qp Optimiser, which is from [SVM toolbox](http://www.isis.ecs.soton.ac.uk/resources/svminfo/).
The qp　Optimiser is only run on widows system. So you can choose 'CD' or 'QP'　algorithm on Linux.


---------
Go into the optimiser directory and type,
  mex -v qp.c pr_loqo.c
  mv qp.mex??? ..

