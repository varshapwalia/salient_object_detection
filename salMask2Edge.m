% Define paths to input and output directories
dataRoot = './dataset/DUTS/DUTS-TR/DUTS-TR-Mask';
outputRoot = './dataset/DUTS/DUTS-TR/DUTS-TR-Mask';
listSet = './dataset/DUTS/DUTS-TR/train';
indexFile = fullfile([listSet '.lst']);

% Open the file containing image IDs
fileID = fopen(indexFile);
imageIDs = textscan(fileID, '%s'); % Read image IDs as strings
imageIDs = imageIDs{1}; % Extract the cell array of strings
fclose(fileID); % Close the file

% Process each image
numImages = length(imageIDs);
for imageIDIndex = 1:numImages
    currentID = imageIDs{imageIDIndex}; % Get the current image ID
    currentID = currentID(1:end-4); % Remove the file extension

    % Read the ground truth mask image
    maskImage = imread(fullfile(dataRoot, [currentID '.png']));
    maskImage = (maskImage > 128); % Threshold the image to create a binary mask
    maskImage = double(maskImage); % Convert binary mask to double precision

    % Calculate gradient to obtain edge map
    [gradientY, gradientX] = gradient(maskImage); % Compute gradient along y-axis and x-axis
    tempEdgeMap = gradientY .* gradientY + gradientX .* gradientX; % Squared magnitude of gradient
    tempEdgeMap(tempEdgeMap ~= 0) = 1; % Threshold to create binary edge map
    edgeMap = uint8(tempEdgeMap * 255); % Convert to 8-bit unsigned integer

    % Save the edge map
    savePath = fullfile(outputRoot, [currentID '_edge.png']); % Construct save path
    imwrite(edgeMap, savePath); % Write edge map to file
end
