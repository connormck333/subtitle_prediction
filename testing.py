import json
import xlsxwriter
from PIL import Image
import os

dataFilePath = "<PATH_TO_JSON_FILE>"
frameFilePath = "<PATH_TO_FRAMES_FOLDER>"
faceshotsFilePath = "<PATH_TO_FACESHOTS_FOLDER>"
frameExtension = ".png"

def outputDataToExcel(data, fileName):

    # try:
        # Create Excel sheet
        imgDimensionsSet = False
        workbook = xlsxwriter.Workbook(fileName + '.xlsx')
        worksheet = workbook.add_worksheet()
        workbook.formats[0].set_font_size(16)
        bold = workbook.add_format({'bold': True})
        bold.set_font_size(18)

        # Output clip data
        clip = data['clip_data']
        worksheet.write('A1', 'Title:', bold)
        worksheet.write('B1', clip['title'])
        worksheet.write('A2', 'Channel Name:', bold)
        worksheet.write('B2', clip['yt_channel'])
        worksheet.write('A3', 'Link to video:', bold)
        worksheet.write('B3', clip['link'])

        # Add titles
        worksheet.write('A4', 'Frame ID', bold)
        worksheet.write('B4', 'Subtitle', bold)
        worksheet.write('C4', 'Frame', bold)
        worksheet.write('D4', 'Speaker Image', bold)
        worksheet.write('E4', 'Speaker ID', bold)
        worksheet.write('F4', 'Count', bold) # No. times spotted in segment
        worksheet.write('G4', 'Biometric Score', bold)
        worksheet.write('H4', 'Start', bold)
        worksheet.write('I4', 'End', bold)

        # Column widths
        worksheet.set_column(0, 0, 20)
        worksheet.set_column(1, 3, 40)
        worksheet.set_column(4, 8, 20)

        for index, current in enumerate(data['scenes']):
            print(index)
            
            worksheet.write('A' + str(index + 5), current['id'])
            worksheet.write('B' + str(index + 5), current['text'])
            worksheet.write('H' + str(index + 5), current['start'])
            worksheet.write('I' + str(index + 5), current['end'])
            
            if len(current['cast'].keys()) != 0:
                worksheet.write('E' + str(index + 5), current['cast']['id'])
                worksheet.write('F' + str(index + 5), current['cast']['count'])
                worksheet.write('G' + str(index + 5), current['cast']['bio_score'])

                # Get image dimensions
                imagePath = frameFilePath + current['cast']['img_code'] + frameExtension
                faceshotPath = faceshotsFilePath + str(index) + frameExtension
                if not imgDimensionsSet:
                    img = Image.open(imagePath)
                    worksheet.set_column(2, 3, img.width / 100 * 5)
                    print(img.width / 100 * 2)
                    worksheet.set_default_row(img.height / 100 * 25)
                    # worksheet.set_row(0, 15)
                    imgDimensionsSet = True
                
                worksheet.insert_image('C' + str(index + 5), imagePath, {'x_scale': 0.25, 'y_scale': 0.25})
                if os.path.exists(faceshotPath):
                    worksheet.insert_image('D' + str(index + 5), faceshotPath, {'x_scale': 0.5, 'y_scale': 0.5})
                else:
                    worksheet.write('D' + str(index + 5), 'Faceshot Not Found')
        
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
    res = outputDataToExcel(jsonData, "Results_1")
    print(res['status'])

    if res['status'] == 'Failed':
        print(res['msg'])