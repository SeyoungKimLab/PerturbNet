addpath('../MATLAB/');
num_modules = 20;
thresh = 1e-3;
write_inferences = 0;

% Reads models
Lambda_y = txt_to_sparse('Lambda_y');
Lambda_z = txt_to_sparse('Lambda_z');
Theta_xy = txt_to_sparse('Theta_xy');
Theta_yz = txt_to_sparse('Theta_yz');
% Reads module clusters
module_labels = dlmread(sprintf('module_labels_%i.txt', num_modules));
% Reads phenotype annotations
pheno_feattypes_fname = 'pheno_featuretypes.txt';
pheno_feattypes_f = fopen(pheno_feattypes_fname);
pheno_fts = textscan(pheno_feattypes_f,'%s'); 
fclose(pheno_feattypes_f);
pheno_featuretypes = pheno_fts{1};
lung_ixs = find(~cellfun(@isempty,strfind(pheno_featuretypes, 'lung')));
blood_ixs = find(~cellfun(@isempty,strfind(pheno_featuretypes, 'blood')));

num_genes = size(Theta_xy, 2);
num_snps = size(Theta_xy, 1);
num_traits = size(Theta_yz, 2);

Sigma_y = inv(Lambda_y);
Sigma_z = inv(Lambda_z);

% Indirect SNP perturbation effects on gene expression levels
B_xy = -Theta_xy * Sigma_y;

% Indirect effects of gene expression levels on clinical phenotypes
B_yz = -Theta_yz * Sigma_z;

% SNP effects on clinical phenotypes
B_xz = B_xy * B_yz;
Sigma_z_given_x = Sigma_z + Sigma_z*Theta_yz'*Sigma_y*Theta_yz*Sigma_z;
% Joint distribution of p(z,y|x)
Lambda_y_given_xz = Lambda_y + Theta_yz * Sigma_z * Theta_yz';
Lambda_zy_given_x = [Lambda_z Theta_yz'; Theta_yz Lambda_y_given_xz];
Theta_zy_given_x = [sparse(num_snps, num_traits) Theta_xy];

% SNP effects on clinical traits mediated by a gene module
for m=1:num_modules
    m_ixs = find(module_labels == m);
    B_xz_m = B_xy(:,m_ixs) * B_yz(m_ixs,:);
end
m0_ixs = find(module_labels == 0);
m_not0_ixs = setdiff((1:num_genes)', m0_ixs);
B_nonsingle = B_xy(:,m_not0_ixs)*B_yz(m_not0_ixs,:);

% Posterior gene network after seeing phenotype data
Lambda_y_given_xz = Lambda_y + Theta_yz * Sigma_z * Theta_yz';


% SNP perturbation effects on gene modules and trait groups
% The effects of SNP i on gene module M
SNP_i_on_module_m_direct = @(i,M) sum(abs(Theta_xy(i,find(module_labels==M))));
SNP_i_on_module_m_indirect = @(i,M) sum(abs(B_xy(i,find(module_labels==M))));
% The effects of gene module M on traits
module_M_on_traits_direct = @(M,trait_ixs) sum(abs(Theta_yz(find(module_labels==M),trait_ixs)));
module_M_on_traits_indirect = @(M,trait_ixs) sum(abs(B_yz(find(module_labels==M),trait_ixs)));
module_M_on_lung = @(M) module_M_on_traits(M, lung_ixs);
module_M_on_blood = @(M) module_M_on_traits(M, blood_ixs);
% The effects of SNP i on traits
SNP_i_on_traits = @(i,trait_ixs) sum(abs(B_xz(i,trait_ixs)));
SNP_i_on_lung = @(i) SNP_i_on_traits(i, lung_ixs);
SNP_i_on_blood = @(i) SNP_i_on_traits(i, blood_ixs);
% The effects of SNP i on traits, mediated by module M
SNP_i_on_traits_mediated_by_M = @(i, trait_ixs, M) ...
    sum(abs(B_xy(i,find(module_labels==M)) * B_yz(find(module_labels==M),trait_ixs)));
SNP_i_on_lung_mediated_by_M = @(i, M) SNP_i_on_traits_mediated_by_M(i, lung_ixs, M);
SNP_i_on_blood_mediated_by_M = @(i, M) SNP_i_on_traits_mediated_by_M(i, blood_ixs, M);

if write_inferences
    B_xy_t = hard_threshold(B_xy, thresh);
    B_yz_t = hard_threshold(B_yz, thresh);
    B_xz_t = hard_threshold(B_xz, thresh);
    sparse_to_txt('B_xy.out', B_xy_t);
    sparse_to_txt('B_yz.out', B_yz_t);
    sparse_to_txt('B_xz.out', B_xz_t);

    Posterior_t = hard_threshold(Posterior, thresh);
    sparse_to_txt('Posterior_yxz.out', Posterior_t);

    sparse_to_txt(sprintf('B_xz_%i.out',m), B_xz_m);

    B_nonsingle_t = hard_threshold(B_nonsingle, thresh);
    sparse_to_txt('B_xz_nonsingle.out', B_nonsingle_t);
end
