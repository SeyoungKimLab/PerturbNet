function struct_to_txt(filename, A)
    fid = fopen(filename, 'w');
    fieldnamesA = fieldnames(A);
    for fieldix=1:length(fieldnamesA)
        fieldname = fieldnamesA{fieldix};
        fieldvalues = getfield(A, fieldname);
        fprintf(fid, '%s ', fieldname);
        fprintf(fid, '%.4f ', fieldvalues);
        fprintf(fid, '\n');
    end
    fclose(fid);
end
