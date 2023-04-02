
target = P10_13;
seg={};

num_image_set = size(target,2);

tot = [];

h=waitbar(0,'Please Wait...');

for i = 1 : num_image_set
   
    seg{i}.slice_num = target{i}.slice_num;
    seg{i}.animal_num = target{i}.animal_num;
    
    seg{i}.th_upd = target{i}.KV4.th;
    seg{i}.th_upd (~target{i}.thresh_upd_ad)=0;
    
    seg{i}.mask = imextendedmin(target{i}.KV4.cl_wie,5000);
    seg{i}.mask = imfill(seg{i}.mask,'holes');
    
    seg{i}.mask(seg{i}.th_upd==0)=0;
    
    [KK,num]=bwlabel(seg{i}.mask);

    for j=1:num
        if sum(sum(KK==j))>50
            seg{i}.mask(KK==j)=0;
        elseif sum(sum(KK==j))<2
            seg{i}.mask(KK==j)=0;
        else
            seg{i}.mask(KK==j)=1;
        end
    end
    
    seg{i}.I_mode=imimposemin(target{i}.KV4.cl_wie,~target{i}.thresh_upd_ad|seg{i}.mask);
    seg{i}.I_m = watershed(seg{i}.I_mode);
    
    [seg{i}.L,num2]=bwlabel(seg{i}.I_m);

    for j=1:num2
        if sum(sum(seg{i}.L==j))>100
            seg{i}.L(seg{i}.L==j)=0;
        elseif sum(sum(seg{i}.L==j))<5
            seg{i}.L(seg{i}.L==j)=0;
        else
            seg{i}.L(seg{i}.L==j)=j;
        end
    end

    seg{i}.fin_mask=im2bw(seg{i}.L,1);
    seg{i}.fin_mask=imclearborder(seg{i}.fin_mask);
    seg{i}.fin_mask=bwareaopen(seg{i}.fin_mask,5);
    seg{i}.fin_perim=bwperim(seg{i}.fin_mask);
    seg{i}.fin_over=imoverlay(target{i}.KV4.raw,seg{i}.fin_perim,[.3 1 .3]);
    
    [AA,BB]=bwlabel(seg{i}.fin_mask);
    if BB~=0
        for j=1:BB
       
           seg{i}.sg_label(j) = j;
        
           seg{i}.sg_gfp{j} = target{i}.GFP.raw (AA==j) ;
           seg{i}.sg_gfp_area(j) = size(seg{i}.sg_gfp{j},1);
           seg{i}.sg_gfp_intensity(j) = sum(seg{i}.sg_gfp{j});
           seg{i}.sg_gfp_avg_int(j) = seg{i}.sg_gfp_intensity(j)/seg{i}.sg_gfp_area(j);
        
           seg{i}.sg_tdt{j} = target{i}.TDT.raw (AA==j) ;
           seg{i}.sg_tdt_area(j) = size(seg{i}.sg_tdt{j},1);
           seg{i}.sg_tdt_intensity(j) = sum(seg{i}.sg_tdt{j});
           seg{i}.sg_tdt_avg_int(j) = seg{i}.sg_tdt_intensity(j)/seg{i}.sg_tdt_area(j);
        
        end
        
        tot=[tot;seg{i}.animal_num*ones(BB,1) seg{i}.slice_num*ones(BB,1) seg{i}.sg_label' seg{i}.sg_gfp_avg_int' seg{i}.sg_tdt_avg_int'];
    end
    
    
    waitbar(i/num_image_set,h,['Please Wait...' ' ' 'Processing..' num2str(i)]);

end


S_P10_13 = seg;
tot_P10_13 = tot;

close(h);

clear num2 i j KK num h num_image_set AA BB seg target tot
