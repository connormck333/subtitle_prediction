import json
import xlsxwriter
from PIL import Image
import io
import os

dataFilePath = "/Users/connormckenzie/Documents/Projects/QUB/CSC1028/PyProject/data_VGG.json"
frameFilePath = "/Users/connormckenzie/Documents/Projects/QUB/CSC1028/YoutubeFiles/images/"
frameExtension = ".png"

def outputDataToExcel(data):

    # try:
        # Create Excel sheet
        imgDimensionsSet = False

        workbook = xlsxwriter.Workbook('results_VGG.xlsx')
        worksheet = workbook.add_worksheet()

        # Add titles
        bold = workbook.add_format({'bold': True})
        worksheet.write('A1', 'Subtitle', bold)
        worksheet.write('B1', 'Image', bold)
        worksheet.write('C1', 'Speaker', bold)
        worksheet.write('D1', 'Count', bold)
        worksheet.write('E1', 'Match Score', bold)
        worksheet.write('F1', 'Start', bold)
        worksheet.write('G1', 'End', bold)

        # Column widths
        worksheet.set_column(0, 0, 40)
        worksheet.set_column(2, 6, 20)

        for i in data.keys():
            index = int(i)
            current = data[str(i)]
            print(index)
            
            worksheet.write('A' + str(index + 2), current['subtitle'])
            worksheet.write('C' + str(index + 2), current['name'])
            worksheet.write('F' + str(index + 2), current['start'])
            worksheet.write('G' + str(index + 2), current['end'])
            
            if 'error' not in current.keys():
                # Get image dimensions
                imagePath = frameFilePath + current['image'] + frameExtension
                if not imgDimensionsSet:
                    img = Image.open(imagePath)
                    worksheet.set_column(1, 1, img.width / 100 * 5)
                    print(img.width / 100 * 2)
                    worksheet.set_default_row(img.height / 100 * 20)
                    worksheet.set_row(0, 15)
                    imgDimensionsSet = True
                
                worksheet.insert_image('B' + str(index + 2), imagePath, {'x_scale': 0.2, 'y_scale': 0.2})
                worksheet.write('D' + str(index + 2), current['frameCount'])
                worksheet.write('E' + str(index + 2), current['frameScore'])
        
        workbook.close()

        return {'status': 'Success'}
    # except Exception as e:
    #     print(e)
    #     return {'status': 'Failed', 'msg': e}
    


def openData(filePath):
    f = open(filePath)

    data = json.load(f)
    
    return data

if __name__ == '__main__':
    jsonData = openData(dataFilePath)
    res = outputDataToExcel(jsonData)
    print(res['status'])

    if res['status'] == 'Failed':
        print(res['msg'])