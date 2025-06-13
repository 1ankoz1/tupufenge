# MATLAB图像分割工具 Readme 文档

## 一、功能介绍
### 1. 工具概述
基于MATLAB GUI开发的交互式图像分割工具，支持加载自然图像、医学图像、工业图像等，并提供四种传统分割算法，实现原始图像与分割结果的可视化对比。工具采用模块化设计，包含界面交互、图像预处理、算法执行和结果显示四大模块，具备良好的鲁棒性和用户体验。

### 2. 核心功能
- **图像加载**：支持JPG、PNG、BMP等常见格式，自动处理透明通道（如PNG），并提供错误提示。
- **分割算法**：实现四种分割方法，支持参数动态调节：
    - 梯度分割法（Sobel、Canny）：边缘检测；
    - 全局阈值法（Otsu）：自动计算二值化阈值；
    - 区域生长法：基于种子点和阈值的区域扩展；
    - 分水岭算法：基于梯度和标记提取的分割，优化过分割问题。
- **可视化展示**：原始图像与分割结果左右并排显示，分水岭结果采用半透明伪彩色叠加，边界黑色高亮。
- **状态反馈**：实时显示操作进度、算法耗时（如“处理完成耗时0.36秒”）和错误信息。

## 二、操作截图说明
### 1. 主界面布局
- **控制面板（顶部，占高度13%）**：
    - 左侧：“加载图像”按钮、分割方法下拉菜单（5种算法）；
    - 右侧：参数输入框（默认值0.2）、“执行分割”按钮、状态栏（显示“就绪”）。
- **图像显示面板（下方，占高度82%）**：
    - 左侧坐标轴：显示原始图像，标题动态更新为文件名（如“原始图像：山.jpg”）；
    - 右侧坐标轴：显示分割结果，标题随算法类型变化（如“分水岭分割结果”）。

### 2. 操作流程截图示例
- **步骤1：加载图像**
    点击“加载图像”按钮，选择文件（如“机械臂.jpg”），左侧显示原始图像，状态栏提示“已加载图像：机械臂.jpg”。
- **步骤2：选择算法并设置参数**
    下拉菜单选择“分水岭算法”，参数输入框设置为0.15（默认0.2）。
- **步骤3：执行分割**
    点击“执行分割”按钮，右侧显示结果，状态栏显示“处理完成(耗时0.36秒)”，分水岭结果以伪彩色叠加显示，边界清晰。

## 三、核心代码注释
### 1. 主函数 imageSegmentationTool：
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

这段 MATLAB 代码构建了一个图像分割工具的 GUI 应用，通过 figure 创建名为 “图像分割工具” 的主窗口，利用 uipanel 划分出控制面板（含加载图像按钮、分割方法下拉菜单、执行分割按钮、参数编辑框等控件 ）和图像显示面板（用 axes 准备原始图与结果图显示区域 ）；定义 guidata 存储图像、控件句柄等关键数据，借助 loadImage 回调实现图像加载（支持多格式、处理透明通道、更新界面显示 ），performSegmentation 回调依据所选分割算法（Sobel、Canny、Otsu、区域生长、分水岭 ）及参数，对加载图像预处理后执行分割，计算耗时并可视化结果，updateStatus 负责实时更新状态栏文本，整体实现了图像加载、算法选择、分割执行与结果展示的交互流程，为用户提供直观的图像分割操作体验。

### 2. 区域生长法优化函数 optimizedRegionGrowing：
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

这段代码实现了一个优化的区域生长算法，用于图像分割。它首先验证输入的阈值参数是否在合理范围（0到1之间），然后将输入的灰度图像归一化到0到1的范围。算法自动选择图像中心区域的最亮点作为种子点，并根据种子点的灰度值动态计算生长阈值。接着，算法从种子点开始，使用4邻域搜索策略逐步扩展区域，直到所有满足条件的邻域像素都被包含进来。为了提高效率，代码使用了预分配内存的队列来管理待处理的像素点。在区域生长完成后，算法通过形态学填充操作去除分割区域内的小孔洞，并将最终的二值化结果转换为uint8类型，以便于后续处理。

### 3. 分水岭算法优化函数 optimizedWatershed：
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

这段代码实现了一个优化的分水岭图像分割算法。它首先对输入的灰度图像进行闭运算以减少过分割，然后通过引导滤波（或双边滤波）进行平滑处理，同时保留边缘信息。接着，代码使用Scharr算子计算梯度幅值，以获取更精确的边缘信息，并通过改进的H-minima变换提取前景标记，同时利用距离变换和分水岭变换提取背景标记。在分水岭变换之前，代码对梯度图像进行修改，强制在标记位置设置最小值，并对梯度图像进行平滑处理以减少噪声影响。分水岭变换完成后，代码通过合并过小的区域、细化边界，并将分割结果以伪彩色叠加到原始图像上，最终生成清晰且准确的分割结果，同时用黑色边界突出显示分割边界，确保输出为uint8类型以便于后续处理。

### 4. 过分割合并辅助函数 mergeOverSegmentedRegions：
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
    
    这段代码实现了一个用于优化图像分割结果的函数 mergeOverSegmentedRegions，主要目的是通过合并过小的区域来减少分水岭算法中常见的过分割问题。函数首先提取每个区域的面积、像素索引、平均灰度等属性，并构建一个邻接矩阵来记录区域之间的连接关系。接着，它通过两轮合并操作来优化分割结果：第一轮合并面积小于设定阈值 minSize 的小区域，第二轮合并那些面积小于所有区域面积中值一半的区域。在每轮合并中，函数会根据灰度相似性和邻接区域面积的综合评分来选择最佳合并目标。通过这种方式，函数能够有效地减少分割结果中的碎片化现象，同时尽量保留图像的主要结构和语义信息，最终返回优化后的标记矩阵。

## 四、操作截图

### 主界面：
![主界面截图](https://media/image1.png)

### 分割结果示例：

#### 自然图像分割：
![自然图像分割结果](https://media/image2.png)

#### 医学图像分割：
![医学图像分割结果](https://media/image3.png)

#### 工业图像分割：
![工业图像分割结果](https://media/image4.png)

## 五、注意事项

- 分水岭算法处理大图像时可能耗时较长  
- 区域生长法的阈值建议设置在0.1-0.5之间  
- 需要安装Image Processing Toolbox  
- 分水岭算法的并行计算需要Parallel Computing Toolbox支持  

## 六、测试图像

工具包中包含以下测试图像：
![自然图像](https://github.com/1ankoz1/tupufenge/blob/main/hill.jpg)
![医学图像](https://github.com/1ankoz1/tupufenge/blob/main/HPV.jpg)
![工业图像](https://github.com/1ankoz1/tupufenge/blob/main/robot_arm.jpg)

## 七、版本信息

- **开发环境**：MATLAB R2023a  
- **最后更新**：2025-06-25  
- **开发者**：曾建明（学号：2022110203）
