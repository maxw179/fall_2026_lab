function saveTiff16(img, filename)
    img = double(img);
    img16 = uint16(max(0, min(65535, img)));
    imwrite(img16, filename, 'tif');
end
