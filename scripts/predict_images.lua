require ('string')
require('cutorch')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('paths')
require('nngraph')

require('libcrnn')
require('utilities')
require('inference')
require('CtcCriterion')
require('DatasetLmdb')
require('LstmLayer')
require('BiRnnJoin')
require('SharedParallelTable')

-- endswith function
function string.ends(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

-- load model
cutorch.setDevice(1)
torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')

print('Loading model...')
local modelDir = '/root/crnn/src/model/crnn_demo'

paths.dofile('/root/crnn/model/crnn_demo/config.lua')
local modelLoadPath = '/root/crnn/src/model/crnn_demo/crnn_demo_model.t7' 

gConfig = getConfig()
gConfig.modelDir = modelDir
gConfig.maxT = 0

local model, criterion = createModel(gConfig)
local snapshot = torch.load(modelLoadPath)
loadModelState(model, snapshot)

model:evaluate()
print(string.format('Model loaded from %s', modelLoadPath))


-- loop over extracted texts
local img_dir = '../data/extracts/'

for file in paths.files(img_dir) do
    local full_img_path = paths.concat(img_dir, file)
    if string.ends(full_img_path, '.png') then 
        print(full_img_path)

	local img = loadAndResizeImage(full_img_path)
	local text, raw = recognizeImageLexiconFree(model, img)
	print(string.format('Recognized text: %s (raw: %s)', text, raw))

	-- Save found string
	local filename = string.gsub(full_img_path, '.png', '.txt')
	local f = assert(io.open(filename, "w"))
	f:write(text)
	f:close()

    end
end





--[[
cutorch.setDevice(1)
torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')

print('Loading model...')
local modelDir = '../model/crnn_demo/'

paths.dofile('/root/crnn/model/crnn_demo/config.lua')
local modelLoadPath = '/root/crnn/src/model/crnn_demo/crnn_demo_model.t7' 

gConfig = getConfig()
gConfig.modelDir = modelDir
gConfig.maxT = 0

local model, criterion = createModel(gConfig)
local snapshot = torch.load(modelLoadPath)
loadModelState(model, snapshot)

model:evaluate()
print(string.format('Model loaded from %s', modelLoadPath))


local imagePath = '../data/images/img_1.jpg'
local img = loadAndResizeImage(imagePath)
local text, raw = recognizeImageLexiconFree(model, img)
print(string.format('Recognized text: %s (raw: %s)', text, raw))
]]
