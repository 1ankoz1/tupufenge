function imageSegmentationTool
    % 创建主界面
    fig = figure('Name', '图像分割工具 ', 'NumberTitle', 'off', ...
        'Position', [100, 100, 900, 650], 'MenuBar', 'none', 'ToolBar', 'none', ...
        'Color', [0.94 0.94 0.94]);
    
    % 添加控制面板
    controlPanel = uipanel('Parent', fig, 'Title', '控制面板', ...
        'Position', [0.02, 0.85, 0.96, 0.13], ...
        'BackgroundColor', [0.94 0.94 0.94]);
    
    % 添加图像显示面板
    imagePanel = uipanel('Parent', fig, 'Title', '图像显示', ...
        'Position', [0.02, 0.02, 0.96, 0.82], ...
        'BackgroundColor', [0.94 0.94 0.94]);
    
    % 添加控件
    uicontrol('Parent', controlPanel, 'Style', 'pushbutton', ...
        'String', '加载图像', 'Position', [20, 20, 100, 30], ...
        'Callback', @loadImage, 'FontWeight', 'bold');
    
    uicontrol('Parent', controlPanel, 'Style', 'text', ...
        'String', '分割方法:', 'Position', [140, 25, 80, 20], ...
        'HorizontalAlignment', 'right', 'BackgroundColor', [0.94 0.94 0.94]);
    
    methodPopup = uicontrol('Parent', controlPanel, 'Style', 'popupmenu', ...
        'String', {'梯度分割法(Sobel)', '梯度分割法(Canny)', '全局阈值法(Otsu)', '区域生长法', '分水岭算法'}, ...
        'Position', [230, 20, 180, 30]);
    
    uicontrol('Parent', controlPanel, 'Style', 'pushbutton', ...
        'String', '执行分割', 'Position', [430, 20, 100, 30], ...
        'Callback', @performSegmentation, 'FontWeight', 'bold');
    
    % 添加参数控制
    uicontrol('Parent', controlPanel, 'Style', 'text', ...
        'String', '参数:', 'Position', [550, 25, 50, 20], ...
        'HorizontalAlignment', 'right', 'BackgroundColor', [0.94 0.94 0.94]);
    
    paramEdit = uicontrol('Parent', controlPanel, 'Style', 'edit', ...
        'String', '0.2', 'Position', [610, 20, 60, 30], ...
        'Tooltip', '区域生长阈值/分水岭参数');
    
    % 添加状态栏
    statusText = uicontrol('Parent', fig, 'Style', 'text', ...
        'String', '就绪', 'Position', [20, 10, 860, 20], ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [0.8 0.8 0.8]);
    
    % 添加轴用于显示图像
    axOriginal = axes('Parent', imagePanel, 'Units', 'normalized', ...
        'Position', [0.05, 0.1, 0.4, 0.8]);
    title(axOriginal, '原始图像');
    axis(axOriginal, 'image');
    axis(axOriginal, 'off');
    
    axResult = axes('Parent', imagePanel, 'Units', 'normalized', ...
        'Position', [0.55, 0.1, 0.4, 0.8]);
    title(axResult, '分割结果');
    axis(axResult, 'image');
    axis(axResult, 'off');
    
    % 存储数据
    handles = struct();
    handles.originalImage = [];
    handles.methodPopup = methodPopup;
    handles.paramEdit = paramEdit;
    handles.axOriginal = axOriginal;
    handles.axResult = axResult;
    handles.statusText = statusText;
    handles.fig = fig;
    guidata(fig, handles);
    
    % 回调函数
    function loadImage(~, ~)
        updateStatus('正在加载图像...');
        
        [filename, pathname] = uigetfile(...
            {'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.gif', '图像文件 (*.jpg, *.png, *.bmp, *.tif, *.gif)'}, ...
            '选择图像文件');
        
        if isequal(filename, 0)
            updateStatus('取消加载图像');
            return;
        end
        
        try
            fullpath = fullfile(pathname, filename);
            img = imread(fullpath);
            
            % 处理透明通道
            if size(img, 3) == 4
                img = img(:,:,1:3); % 去除alpha通道
            end
            
            handles = guidata(fig);
            handles.originalImage = img;
            guidata(fig, handles);
            
            % 显示图像
            axes(handles.axOriginal);
            imshow(img);
            title(handles.axOriginal, ['原始图像: ' filename], 'Interpreter', 'none');
            
            % 清除之前的结果
            cla(handles.axResult);
            title(handles.axResult, '分割结果');
            
            updateStatus(['已加载图像: ' filename]);
        catch ME
            errordlg(['加载图像失败: ' ME.message], '错误');
            updateStatus('图像加载失败');
        end
    end

    function performSegmentation(~, ~)
        handles = guidata(fig);
        if isempty(handles.originalImage)
            errordlg('请先加载图像!', '错误');
            updateStatus('错误: 未加载图像');
            return;
        end
        
        img = handles.originalImage;
        method = get(handles.methodPopup, 'Value');
        paramStr = get(handles.paramEdit, 'String');
        
        try
            % 验证参数
            param = str2double(paramStr);
            if isnan(param)
                error('参数必须是数值');
            end
            
            updateStatus('正在处理图像...');
            drawnow; % 更新UI
            
            % 记录处理时间
            tic;
            
            % 根据图像类型预处理
            if size(img, 3) == 3
                grayImg = rgb2gray(img);
            else
                grayImg = img;
            end
            
            % 执行分割
            switch method
                case 1 % Sobel
                    edgeImg = edge(grayImg, 'sobel');
                    result = uint8(255 * edgeImg);
                    resultTitle = 'Sobel边缘检测结果';
                    
                case 2 % Canny
                    [~, threshold] = edge(grayImg, 'canny');
                    edgeImg = edge(grayImg, 'canny', threshold * 0.5);
                    result = uint8(255 * edgeImg);
                    resultTitle = 'Canny边缘检测结果';
                    
                case 3 % Otsu
                    level = graythresh(grayImg);
                    result = imbinarize(grayImg, level);
                    resultTitle = 'Otsu阈值分割结果';
                    
                case 4 % 区域生长法
                    result = optimizedRegionGrowing(grayImg, param);
                    resultTitle = sprintf('区域生长结果 (阈值=%.2f)', param);
                    
                case 5 % 分水岭算法
                    result = optimizedWatershed(grayImg, param);
                    resultTitle = '分水岭分割结果';
            end
            
            % 显示处理时间
            elapsedTime = toc;
            
            % 显示结果
            axes(handles.axResult);
            if method == 5 % 分水岭算法特殊处理
                imshow(result);
            else
                imshow(result, []);
            end
            title(handles.axResult, resultTitle);
            
            updateStatus(sprintf('处理完成 (耗时 %.2f秒)', elapsedTime));
            
        catch ME
            errordlg(['分割过程中出错: ' ME.message], '错误');
            updateStatus(['错误: ' ME.message]);
        end
    end

    function updateStatus(message)
        handles = guidata(fig);
        set(handles.statusText, 'String', ['状态: ' message]);
        drawnow;
    end
end

%% 区域生长法
function result = optimizedRegionGrowing(grayImg, thresholdRatio)
    % 输入验证
    if thresholdRatio <= 0 || thresholdRatio >= 1
        error('阈值参数应在0-1之间');
    end
    
    % 标准化图像
    grayImg = mat2gray(grayImg);
    
    % 自动选择种子点 - 使用图像中心区域的平均值
    [h, w] = size(grayImg);
    centerRegion = grayImg(round(h/4):round(3*h/4), round(w/4):round(3*w/4));
    [~, idx] = max(centerRegion(:));
    [y, x] = ind2sub(size(centerRegion), idx);
    x = x + round(w/4) - 1;
    y = y + round(h/4) - 1;
    
    % 计算动态阈值
    seedValue = grayImg(y, x);
    threshold = thresholdRatio * seedValue;
    
    % 初始化
    output = false(size(grayImg));
    output(y, x) = true;
    
    % 使用更高效的队列实现
    queue = zeros(numel(grayImg), 2); % 预分配内存
    queue(1,:) = [x, y];
    queueSize = 1;
    queuePointer = 1;
    
    % 4邻域偏移量 (比8邻域更不容易泄漏)
    neighborOffsets = [-1 0; 1 0; 0 -1; 0 1];
    
    while queuePointer <= queueSize
        currentPoint = queue(queuePointer,:);
        queuePointer = queuePointer + 1;
        
        % 检查所有邻域
        for k = 1:size(neighborOffsets, 1)
            neighbor = currentPoint + neighborOffsets(k,:);
            
            % 检查边界
            if neighbor(1) >= 1 && neighbor(1) <= w && ...
               neighbor(2) >= 1 && neighbor(2) <= h
               
               if ~output(neighbor(2), neighbor(1))
                   pixelValue = grayImg(neighbor(2), neighbor(1));
                   if abs(pixelValue - seedValue) <= threshold
                       output(neighbor(2), neighbor(1)) = true;
                       queueSize = queueSize + 1;
                       queue(queueSize,:) = neighbor;
                   end
               end
            end
        end
    end
    
    % 后处理 - 填充小孔洞
    output = imfill(output, 'holes');
    result = uint8(output * 255);
end

%% 分水岭算法
function result = optimizedWatershed(grayImg, h_param)
grayImg = imclose(grayImg, strel('disk', 5)); % 减少初始过分割
    % 参数设置
    if nargin < 2
        h_param = 0.1; % 默认H-minima参数
    end
    
    % 转换为double类型并归一化
    grayImg = im2double(grayImg);
    
    % 1. 预处理阶段 ------------------------------------------------------
    % 使用引导滤波进行边缘保持平滑
    if exist('imguidedfilter', 'file')
        grayImg = imguidedfilter(grayImg);
    else
        % 如果没有图像处理工具箱，使用双边滤波
        grayImg = imbilatfilt(grayImg);
    end
    
    % 2. 梯度计算阶段 ---------------------------------------------------
    % 使用Scharr算子计算更精确的梯度
    [gx, gy] = imgradientxy(grayImg, 'sobel');
    gradMag = sqrt(gx.^2 + gy.^2);
    
    % 3. 标记提取阶段 ---------------------------------------------------
    % 3.1 前景标记提取（使用改进的H-minima变换）
    h = h_param * max(grayImg(:)); % 动态计算h值
    Ihm = imhmin(grayImg, h);
    fgm = imregionalmin(Ihm);
    
    % 去除太小的区域和边界上的标记
    fgm = bwareaopen(fgm, 20);
    fgm = imclearborder(fgm);
    
    % 3.2 背景标记提取（改进的距离变换方法）
    bw = imbinarize(grayImg, 'adaptive', 'Sensitivity', 0.4);
    D = bwdist(bw);
    D = -D; % 准备分水岭变换
    D(~bw) = -Inf;
    
    % 使用分水岭初步获取背景标记
    Ld = watershed(D);
    bgm = Ld == 0;
    
    % 3.3 未知区域定义
    fgm2 = imdilate(fgm, strel('disk', 2)); % 适度扩大前景标记
    unknown = ~(fgm2 | bgm);
    
    % 4. 修改梯度图像 --------------------------------------------------
    % 4.1 在标记位置强制设置最小值
    markers = bwlabel(fgm);
    markers(unknown) = 0; % 未知区域设为0
    
    % 4.2 平滑梯度图像但保持边缘
    gradMag2 = imgaussfilt(gradMag, 0.5);
    gradMag2 = imimposemin(gradMag2, markers | bgm);
    
    % 5. 执行分水岭变换 -----------------------------------------------
    L = watershed(gradMag2);
    
    % 6. 后处理阶段 ---------------------------------------------------
    % 6.1 合并过分割的小区域
    L = mergeOverSegmentedRegions(L, grayImg, 100);
    
    % 6.2 边界精细化
    boundaries = (L == 0);
    boundaries = bwmorph(boundaries, 'thin', Inf); % 细化边界
    
    % 6.3 创建彩色结果（边界更清晰）
    coloredLabels = label2rgb(L, 'jet', 'w', 'shuffle');
    
    % 将原始灰度图像转换为RGB（如果是灰度图）
    if size(grayImg, 3) == 1
        originalRGB = repmat(mat2gray(grayImg), [1 1 3]);
    else
        originalRGB = grayImg;
    end
    
    % 设置透明度 (0.3表示30%透明度)
    alpha = 0.3;
    
    % 将伪彩色标签叠加到原始图像上
    result = originalRGB * (1-alpha) + im2double(coloredLabels) * alpha;
    
    % 添加黑色边界
    boundaries = (L == 0);
    result = imoverlay(result, boundaries, [0 0 0]); % 用黑色强调边界
    
    % 确保输出是uint8类型
    result = im2uint8(result);

end

%% 辅助函数：合并过分割的小区域
function L = mergeOverSegmentedRegions(L, grayImg, minSize)
    stats = regionprops(L, grayImg, 'Area', 'PixelIdxList', 'MeanIntensity', 'BoundingBox');
    
    % 创建邻接矩阵（使用8邻域连接）
    adjMat = zeros(max(L(:)));
    [y, x] = find(L == 0); % 边界像素
    
    % 8邻域偏移量
    offsets = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
    
    for k = 1:length(y)
        % 获取边界像素的8邻域区域
        neighbors = [];
        for m = 1:size(offsets, 1)
            ny = y(k) + offsets(m,1);
            nx = x(k) + offsets(m,2);
            if ny >= 1 && ny <= size(L,1) && nx >= 1 && nx <= size(L,2)
                if L(ny, nx) > 0
                    neighbors = [neighbors; L(ny, nx)];
                end
            end
        end
        neighbors = unique(neighbors);
        
        % 填充邻接矩阵
        for i = 1:length(neighbors)
            for j = i+1:length(neighbors)
                adjMat(neighbors(i), neighbors(j)) = adjMat(neighbors(i), neighbors(j)) + 1;
                adjMat(neighbors(j), neighbors(i)) = adjMat(neighbors(j), neighbors(i)) + 1;
            end
        end
    end
    
    % 第一次合并：基于面积的小区域合并
    smallRegions = find([stats.Area] < minSize);
    merged = false(size(smallRegions));
    
    for i = 1:length(smallRegions)
        if merged(i), continue; end
        
        idx = smallRegions(i);
        neighbors = find(adjMat(idx,:));
        neighbors = setdiff(neighbors, smallRegions(merged)); % 排除已合并区域
        
        if isempty(neighbors)
            continue;
        end
        
        % 基于面积和灰度相似性选择最佳邻接区域
        currentValue = stats(idx).MeanIntensity;
        neighborValues = [stats(neighbors).MeanIntensity];
        neighborAreas = [stats(neighbors).Area];
        
        % 综合评分：相似性(70%) + 大区域优先(30%)
        scores = 0.7*(1 - abs(neighborValues - currentValue)/max(grayImg(:))) + ...
                 0.3*(neighborAreas/max(neighborAreas));
        [~, bestIdx] = max(scores);
        bestNeighbor = neighbors(bestIdx);
        
        % 合并区域
        L(stats(idx).PixelIdxList) = bestNeighbor;
        merged(i) = true;
        
        % 更新邻接矩阵（将合并区域的连接数加到目标区域）
        adjMat(bestNeighbor,:) = adjMat(bestNeighbor,:) + adjMat(idx,:);
        adjMat(:,bestNeighbor) = adjMat(:,bestNeighbor) + adjMat(:,idx);
    end
    
    % 第二次合并：检查剩余小区域（即使大于minSize但仍然是相对小的区域）
    statsNew = regionprops(L, grayImg, 'Area', 'PixelIdxList', 'MeanIntensity');
    allAreas = [statsNew.Area];
    medianArea = median(allAreas);
    
    for i = 1:length(statsNew)
        if statsNew(i).Area < 0.5*medianArea % 合并小于中值面积一半的区域
            neighbors = find(adjMat(i,:));
            if isempty(neighbors), continue; end
            
            % 选择共享边界最多的邻接区域
            [~, bestIdx] = max(adjMat(i,neighbors));
            bestNeighbor = neighbors(bestIdx);
            L(statsNew(i).PixelIdxList) = bestNeighbor;
            % 在mergeOverSegmentedRegions函数最后添加：
        end
    end
end