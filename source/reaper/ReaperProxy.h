#pragma once

#include <exception>
#include <juce_core/juce_core.h>

#include "IReaperHostApplication.h"

struct ReaperProxy
{
    class Missing : public std::exception
    {
    public:
        Missing (const char* fn) : functionName (fn) {}
        const char* what() const noexcept { return functionName; }
    private:
        const char* functionName;
    };

    ReaperProxy() {}

    void load(reaper::IReaperHostApplication* reaperHost)
    {
        if (reaperHost == nullptr)
            return;

        hasAddMediaItemToTrack = reaperHost->getReaperApi("AddMediaItemToTrack");
        hasAddProjectMarker2 = reaperHost->getReaperApi("AddProjectMarker2");
        hasAddTakeToMediaItem = reaperHost->getReaperApi("AddTakeToMediaItem");
        hasCountMediaItems = reaperHost->getReaperApi("CountMediaItems");
        hasCountSelectedTracks = reaperHost->getReaperApi("CountSelectedTracks");
        hasGetActiveTake = reaperHost->getReaperApi("GetActiveTake");
        hasGetCursorPositionEx = reaperHost->getReaperApi("GetCursorPositionEx");
        hasGetItemStateChunk = reaperHost->getReaperApi("GetItemStateChunk");
        hasGetLastTouchedTrack = reaperHost->getReaperApi("GetLastTouchedTrack");
        hasGetMediaItem = reaperHost->getReaperApi("GetMediaItem");
        hasGetMediaItemTake_Source = reaperHost->getReaperApi("GetMediaItemTake_Source");
        hasGetMediaSourceFileName = reaperHost->getReaperApi("GetMediaSourceFileName");
        hasGetSelectedMediaItem = reaperHost->getReaperApi("GetSelectedMediaItem");
        hasGetSelectedTrack = reaperHost->getReaperApi("GetSelectedTrack");
        hasGetSetMediaItemInfo = reaperHost->getReaperApi("GetSetMediaItemInfo");
        hasGetSetMediaTrackInfo_String = reaperHost->getReaperApi("GetSetMediaTrackInfo_String");
        hasGetTrack = reaperHost->getReaperApi("GetTrack");
        hasInsertTrackInProject = reaperHost->getReaperApi("InsertTrackInProject");
        hasMain_OnCommandEx = reaperHost->getReaperApi("Main_OnCommandEx");
        hasPreventUIRefresh = reaperHost->getReaperApi("PreventUIRefresh");
        hasSelectAllMediaItems = reaperHost->getReaperApi("SelectAllMediaItems");
        hasSetEditCurPos2 = reaperHost->getReaperApi("SetEditCurPos2");
        hasSetItemStateChunk = reaperHost->getReaperApi("SetItemStateChunk");
        hasSetMediaItemLength = reaperHost->getReaperApi("SetMediaItemLength");
        hasSetMediaItemPosition = reaperHost->getReaperApi("SetMediaItemPosition");
        hasSetOnlyTrackSelected = reaperHost->getReaperApi("SetOnlyTrackSelected");
        hasShowConsoleMsg = reaperHost->getReaperApi("ShowConsoleMsg");
        hasUndo_BeginBlock2 = reaperHost->getReaperApi("Undo_BeginBlock2");
        hasUndo_EndBlock2 = reaperHost->getReaperApi("Undo_EndBlock2");
    }

    class MediaItem;
    class MediaTrack;
    class MediaTake;
    class ReaProject;

    static constexpr ReaProject* activeProject = nullptr;

    JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wgnu-zero-variadic-macro-arguments")

#define REAPER_CALL(functionName, typ, ...) \
    if (has##functionName) \
        return reinterpret_cast<typ> (has##functionName) (__VA_ARGS__); \
    else \
        throw Missing (#functionName);

    void* hasAddMediaItemToTrack = nullptr;
    MediaItem* AddMediaItemToTrack (MediaTrack* tr)
    {
        REAPER_CALL(AddMediaItemToTrack, MediaItem* (*) (MediaTrack*), tr)
    }

    void* hasAddProjectMarker2 = nullptr;
    int AddProjectMarker2 (ReaProject* proj, bool isrgn, double pos, double rgnend, const char* name, int wantidx, int color)
    {
        REAPER_CALL(AddProjectMarker2, int (*) (ReaProject*, bool, double, double, const char*, int, int), proj, isrgn, pos, rgnend, name, wantidx, color)
    }

    void* hasAddTakeToMediaItem = nullptr;
    MediaTake* AddTakeToMediaItem (MediaItem* item)
    {
        REAPER_CALL(AddTakeToMediaItem, MediaTake* (*) (MediaItem*), item)
    }

    void* hasCountMediaItems = nullptr;
    int CountMediaItems (ReaProject* proj)
    {
        REAPER_CALL(CountMediaItems, int (*) (ReaProject*), proj)
    }

    void* hasCountSelectedTracks = nullptr;
    int CountSelectedTracks (ReaProject* proj)
    {
        REAPER_CALL(CountSelectedTracks, int (*) (ReaProject*), proj)
    }

    void* hasGetActiveTake = nullptr;
    MediaTake* GetActiveTake (MediaItem* item)
    {
        REAPER_CALL(GetActiveTake, MediaTake* (*) (MediaItem*), item)
    }

    void* hasGetCursorPositionEx = nullptr;
    double GetCursorPositionEx (ReaProject* proj)
    {
        REAPER_CALL(GetCursorPositionEx, double (*) (ReaProject*), proj)
    }

    void* hasGetItemStateChunk = nullptr;
    bool GetItemStateChunk (MediaItem* item, char* strNeedBig, int strNeedBig_sz, bool isundoOptional)
    {
        REAPER_CALL(GetItemStateChunk, bool (*) (MediaItem*, char*, int, bool), item, strNeedBig, strNeedBig_sz, isundoOptional)
    }

    void* hasGetLastTouchedTrack = nullptr;
    MediaTrack* GetLastTouchedTrack ()
    {
        REAPER_CALL(GetLastTouchedTrack, MediaTrack* (*) ())
    }

    void* hasGetMediaItem = nullptr;
    MediaItem* GetMediaItem (ReaProject* proj, int itemidx)
    {
        REAPER_CALL(GetMediaItem, MediaItem* (*) (ReaProject*, int), proj, itemidx)
    }

    class PCM_source;

    void* hasGetMediaItemTake_Source = nullptr;
    PCM_source* GetMediaItemTake_Source (MediaTake* take)
    {
        REAPER_CALL(GetMediaItemTake_Source, PCM_source* (*) (MediaTake*), take)
    }

    void* hasGetMediaSourceFileName = nullptr;
    void GetMediaSourceFileName (PCM_source* source, char* filenamebuf, int filenamebuf_sz)
    {
        REAPER_CALL(GetMediaSourceFileName, void (*) (PCM_source*, char*, int), source, filenamebuf, filenamebuf_sz)
    }

    void* hasGetSelectedMediaItem = nullptr;
    MediaItem* GetSelectedMediaItem (ReaProject* proj, int selitem)
    {
        REAPER_CALL(GetSelectedMediaItem, MediaItem* (*) (ReaProject*, int), proj, selitem)
    }

    void* hasGetSelectedTrack = nullptr;
    MediaTrack* GetSelectedTrack (ReaProject* proj, int seltrackidx)
    {
        REAPER_CALL(GetSelectedTrack, MediaTrack* (*) (ReaProject*, int), proj, seltrackidx)
    }

    void* hasGetSetMediaItemInfo = nullptr;
    double GetSetMediaItemInfo (MediaItem* item, const char* parmname, double setNewValue)
    {
        REAPER_CALL(GetSetMediaItemInfo, double (*) (MediaItem*, const char*, double), item, parmname, setNewValue)
    }

    void* hasGetSetMediaTrackInfo_String = nullptr;
    bool GetSetMediaTrackInfo_String (MediaTrack* tr, const char* parmname, char* stringNeedBig, bool setNewValue)
    {
        REAPER_CALL(GetSetMediaTrackInfo_String, bool (*) (MediaTrack*, const char*, char*, bool), tr, parmname, stringNeedBig, setNewValue)
    }

    void* hasGetTrack = nullptr;
    MediaTrack* GetTrack (ReaProject* proj, int trackidx)
    {
        REAPER_CALL(GetTrack, MediaTrack* (*) (ReaProject*, int), proj, trackidx)
    }

    void* hasInsertTrackInProject = nullptr;
    MediaTrack* InsertTrackInProject (ReaProject* proj, int idx, int flags)
    {
        REAPER_CALL(InsertTrackInProject, MediaTrack* (*) (ReaProject*, int, int), proj, idx, flags)
    }

    void* hasMain_OnCommandEx = nullptr;
    void Main_OnCommandEx (int command, int flag, ReaProject* proj)
    {
        REAPER_CALL(Main_OnCommandEx, void (*) (int, int, ReaProject*), command, flag, proj)
    }

    void* hasPreventUIRefresh = nullptr;
    void PreventUIRefresh (int state)
    {
        REAPER_CALL(PreventUIRefresh, void (*) (int), state)
    }

    void* hasSelectAllMediaItems = nullptr;
    void SelectAllMediaItems (ReaProject* proj, bool selected)
    {
        REAPER_CALL(SelectAllMediaItems, void (*) (ReaProject*, bool), proj, selected)
    }

    void* hasSetEditCurPos2 = nullptr;
    void SetEditCurPos2 (ReaProject* proj, double time, bool moveview, bool seekplay)
    {
        REAPER_CALL(SetEditCurPos2, void (*) (ReaProject*, double, bool, bool), proj, time, moveview, seekplay)
    }

    void* hasSetItemStateChunk = nullptr;
    bool SetItemStateChunk (MediaItem* item, const char* str, bool isundoOptional)
    {
        REAPER_CALL(SetItemStateChunk, bool (*) (MediaItem*, const char*, bool), item, str, isundoOptional)
    }

    void* hasSetMediaItemLength = nullptr;
    void SetMediaItemLength (MediaItem* item, double length, bool refreshUI)
    {
        REAPER_CALL(SetMediaItemLength, void (*) (MediaItem*, double, bool), item, length, refreshUI)
    }

    void* hasSetMediaItemPosition = nullptr;
    void SetMediaItemPosition (MediaItem* item, double position, bool refreshUI)
    {
        REAPER_CALL(SetMediaItemPosition, void (*) (MediaItem*, double, bool), item, position, refreshUI)
    }

    void* hasSetOnlyTrackSelected = nullptr;
    void SetOnlyTrackSelected (MediaTrack* track)
    {
        REAPER_CALL(SetOnlyTrackSelected, void (*) (MediaTrack*), track)
    }

    void* hasShowConsoleMsg = nullptr;
    void ShowConsoleMsg (const char* msg)
    {
        REAPER_CALL(ShowConsoleMsg, void (*) (const char*), msg)
    }

    void* hasUndo_BeginBlock2 = nullptr;
    void Undo_BeginBlock2(ReaProject* proj)
    {
        REAPER_CALL(Undo_BeginBlock2, void (*) (ReaProject*), proj)
    }

    void* hasUndo_EndBlock2 = nullptr;
    void Undo_EndBlock2 (ReaProject* proj, const char* descchange, int extraflags)
    {
        REAPER_CALL(Undo_EndBlock2, void (*) (ReaProject*, const char*, int), proj, descchange, extraflags)
    }

#undef REAPER_CALL

    JUCE_END_IGNORE_WARNINGS_GCC_LIKE

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ReaperProxy)
};
