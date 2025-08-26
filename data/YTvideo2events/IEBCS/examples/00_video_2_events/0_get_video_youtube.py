# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 22:01:00 2025
@author: c3055922

Adapted from IEBCS example
    This script gets a video from YouTube
"""


import yt_dlp

#URLS = ['https://www.youtube.com/watch?v=RtUQ_pz5wlo'] 
URLS = ['https://www.youtube.com/watch?v=tP2TWhDVPVY']


def format_selector(ctx):
    """ Select the best video and the best audio that won't result in an mkv.
    NOTE: This is just an example and does not handle all cases """

    # formats are already sorted worst to best
    formats = ctx.get('formats')[::-1]

    # acodec='none' means there is no audio
    best_video = next(f for f in formats
                      if f['vcodec'] != 'none' and f['acodec'] == 'none')


    # These are the minimum required fields for a merged format
    yield {
        'format_id': f'{best_video["format_id"]}',
        'ext': best_video['ext'],
        'requested_formats': [best_video],
        # Must be + separated list of protocols
        'protocol': f'{best_video["protocol"]}',
    }


ydl_opts = {
    'format': format_selector,
    #'outtmpl': {'default': '../../data/hummingbird_video.%(ext)s',
    'outtmpl': {'default': '../../data/jumpfrog_video.%(ext)s'
                },
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(URLS)