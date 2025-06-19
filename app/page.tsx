"use client";
import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import * as z from "zod";
import { Heading } from "@/components/heading";
import { zodResolver } from "@hookform/resolvers/zod";
import { Form, FormControl, FormField, FormItem } from "@/components/ui/form";
import { useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Loader } from "@/components/loader";
import { 
  MessageCircle, 
  GitBranch, 
  Send, 
  Trash2, 
  Copy, 
  ExternalLink,
  CheckCircle,
  AlertCircle,
  Database,
  Clock,
  Sparkles
} from "lucide-react";

interface Message {
  role: "user" | "bot";
  content: string;
  timestamp: Date;
  sources?: string[];
}

interface SessionInfo {
  session_id: string;
  conversation_length: number;
  created_at: string;
}

const ConversationPage = () => {
  const [indexResponse, setIndexResponse] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);
  const [useEnhanced, setUseEnhanced] = useState(true);
  const [repositoryCount, setRepositoryCount] = useState<number>(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const indexForm = useForm({
    resolver: zodResolver(
      z.object({
        github_url: z.string().url({ message: "Please enter a valid GitHub URL." }),
      })
    ),
  });
  
  const chatForm = useForm({
    resolver: zodResolver(
      z.object({
        message: z.string().min(1, { message: "Message is required." }),
      })
    ),
  });

  // Generate session ID on component mount
  useEffect(() => {
    const newSessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    loadDocumentCount();
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const loadDocumentCount = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/docs-count");
      const data = await response.json();
      setRepositoryCount(data.total_chunks || 0);
    } catch (error) {
      console.error("Error loading document count:", error);
    }
  };

  const onIndexSubmit = async (data: any) => {
    setIsIndexing(true);
    setIndexResponse("");
    
    try {
      console.log("Indexing GitHub repo:", data.github_url);
      const endpoint = `http://localhost:8000/api/index`;
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ github_url: data.github_url }),
      });

      const json = await response.json();
      console.log(json);
      
      if (response.ok) {
        setIndexResponse(`✅ Repository indexed successfully! Processed ${json.documents_processed} documents.`);
        loadDocumentCount(); // Refresh document count
        indexForm.reset();
      } else {
        setIndexResponse(`❌ Failed to index repository: ${json.detail || "Unknown error"}`);
      }
    } catch (error) {
      console.error("Error indexing repository: ", error);
      setIndexResponse("❌ Failed to index repository. Please check the URL and try again.");
    } finally {
      setIsIndexing(false);
    }
  };

  const onChatSubmit = async (data: any) => {
    setIsLoading(true);
    
    // Add user message immediately
    const userMessage: Message = {
      role: "user",
      content: data.message,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);
    
    try {
      const endpoint = useEnhanced ? "http://localhost:8000/api/enhanced-chat" : "http://localhost:8000/api/chat";
      const requestBody = useEnhanced 
        ? { message: data.message, session_id: sessionId }
        : { message: data.message };
      
      const response = await axios.post(endpoint, requestBody);
      
      const botMessage: Message = {
        role: "bot",
        content: response.data.answer,
        timestamp: new Date(),
        sources: response.data.sources || [],
      };
      
      setMessages(prev => [...prev, botMessage]);
      
      // Update session info if using enhanced mode
      if (useEnhanced && response.data.session_id) {
        setSessionInfo({
          session_id: response.data.session_id,
          conversation_length: response.data.conversation_length || 0,
          created_at: new Date().toISOString(),
        });
      }
      
    } catch (error) {
      console.error("Error in chat: ", error);
      const errorMessage: Message = {
        role: "bot",
        content: "❌ Sorry, I encountered an error processing your message. Please try again.",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    }
    
    setIsLoading(false);
    chatForm.reset();
  };

  const clearConversation = () => {
    setMessages([]);
    setSessionInfo(null);
    const newSessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 lg:px-8 py-6">
          <Heading
            title="🤖 RAG Chat with GitHub Repositories"
            description="Index any GitHub repository and have intelligent conversations about its codebase using AI."
          />
          
          {/* Stats Bar */}
          <div className="mt-4 flex flex-wrap gap-4 text-sm">
            <div className="flex items-center gap-2 bg-blue-50 px-3 py-1 rounded-full">
              <Database className="w-4 h-4 text-blue-600" />
              <span className="text-blue-700">{repositoryCount} documents indexed</span>
            </div>
            {sessionInfo && (
              <div className="flex items-center gap-2 bg-green-50 px-3 py-1 rounded-full">
                <MessageCircle className="w-4 h-4 text-green-600" />
                <span className="text-green-700">{sessionInfo.conversation_length} exchanges</span>
              </div>
            )}
            <div className="flex items-center gap-2 bg-purple-50 px-3 py-1 rounded-full">
              <Sparkles className="w-4 h-4 text-purple-600" />
              <span className="text-purple-700">{useEnhanced ? "Enhanced" : "Basic"} mode</span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Sidebar - Repository Management */}
          <div className="lg:col-span-1 space-y-6">
            
            {/* Mode Toggle */}
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-purple-600" />
                Chat Mode
              </h3>
              <div className="space-y-3">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="radio"
                    name="mode"
                    checked={useEnhanced}
                    onChange={() => setUseEnhanced(true)}
                    className="text-purple-600"
                  />
                  <div>
                    <div className="font-medium">Enhanced Mode</div>
                    <div className="text-sm text-gray-500">Conversation memory & context</div>
                  </div>
                </label>
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="radio"
                    name="mode"
                    checked={!useEnhanced}
                    onChange={() => setUseEnhanced(false)}
                    className="text-blue-600"
                  />
                  <div>
                    <div className="font-medium">Basic Mode</div>
                    <div className="text-sm text-gray-500">Stateless responses</div>
                  </div>
                </label>
              </div>
            </div>

            {/* Repository Index Form */}
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <GitBranch className="w-5 h-5 text-blue-600" />
                Index Repository
              </h3>
              
              <Form {...indexForm}>
                <form onSubmit={indexForm.handleSubmit(onIndexSubmit)} className="space-y-4">
                  <FormField
                    name="github_url"
                    render={({ field }) => (
                      <FormItem>
                        <FormControl>
                          <Input 
                            {...field} 
                            placeholder="https://github.com/user/repo" 
                            className="w-full"
                          />
                        </FormControl>
                      </FormItem>
                    )}
                  />
                  <Button 
                    type="submit" 
                    disabled={isIndexing}
                    className="w-full bg-blue-600 hover:bg-blue-700"
                  >
                    {isIndexing ? (
                      <>
                        <Loader />
                        <span className="ml-2">Indexing...</span>
                      </>
                    ) : (
                      <>
                        <GitBranch className="w-4 h-4 mr-2" />
                        Index Repository
                      </>
                    )}
                  </Button>
                </form>
              </Form>
              
              {indexResponse && (
                <div className={`mt-4 p-3 rounded-lg text-sm ${
                  indexResponse.includes('✅') 
                    ? 'bg-green-50 text-green-700 border border-green-200' 
                    : 'bg-red-50 text-red-700 border border-red-200'
                }`}>
                  {indexResponse}
                </div>
              )}
            </div>

            {/* Session Info */}
            {useEnhanced && sessionInfo && (
              <div className="bg-white rounded-xl shadow-sm border p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Clock className="w-5 h-5 text-green-600" />
                  Session Info
                </h3>
                <div className="space-y-2 text-sm">
                  <div><strong>Session ID:</strong> {sessionInfo.session_id.slice(-8)}</div>
                  <div><strong>Messages:</strong> {sessionInfo.conversation_length}</div>
                  <Button 
                    onClick={clearConversation}
                    variant="outline"
                    size="sm"
                    className="w-full mt-3"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Clear Conversation
                  </Button>
                </div>
              </div>
            )}
          </div>

          {/* Main Chat Area */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm border h-[70vh] flex flex-col">
              
              {/* Chat Header */}
              <div className="p-6 border-b bg-gray-50 rounded-t-xl">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <MessageCircle className="w-5 h-5 text-blue-600" />
                  Chat Interface
                  {useEnhanced && <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full">Enhanced</span>}
                </h3>
              </div>

              {/* Messages Container */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.length === 0 && !isLoading && (
                  <div className="text-center text-gray-500 py-12">
                    <MessageCircle className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                    <p>Start a conversation by typing a message below.</p>
                    <p className="text-sm mt-2">Ask about repository structure, code explanations, or implementation details.</p>
                  </div>
                )}

                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex gap-4 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div className={`max-w-[80%] rounded-lg p-4 ${
                      message.role === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-gray-100 text-gray-800"
                    }`}>
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1">
                          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                          {message.sources && message.sources.length > 0 && (
                            <div className="mt-3 pt-3 border-t border-gray-200">
                              <p className="text-xs font-medium mb-2">Sources:</p>
                              {message.sources.slice(0, 2).map((source, idx) => (
                                <div key={idx} className="text-xs bg-white bg-opacity-20 rounded p-2 mb-1">
                                  {source.slice(0, 100)}...
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                        <Button
                          onClick={() => copyToClipboard(message.content)}
                          variant="ghost"
                          size="sm"
                          className={`ml-2 ${message.role === "user" ? "text-white hover:bg-blue-700" : "text-gray-600 hover:bg-gray-200"}`}
                        >
                          <Copy className="w-3 h-3" />
                        </Button>
                      </div>
                      <div className={`text-xs mt-2 ${message.role === "user" ? "text-blue-100" : "text-gray-500"}`}>
                        {message.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 rounded-lg p-4 flex items-center gap-3">
                      <Loader />
                      <span className="text-gray-600">AI is thinking...</span>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>

              {/* Chat Input */}
              <div className="p-6 border-t bg-gray-50 rounded-b-xl">
                <Form {...chatForm}>
                  <form onSubmit={chatForm.handleSubmit(onChatSubmit)} className="flex gap-3">
                    <FormField
                      name="message"
                      render={({ field }) => (
                        <FormItem className="flex-1">
                          <FormControl>
                            <Input 
                              {...field} 
                              placeholder="Ask about the repository..." 
                              className="border-gray-300 focus:border-blue-500"
                              disabled={isLoading}
                            />
                          </FormControl>
                        </FormItem>
                      )}
                    />
                    <Button 
                      type="submit" 
                      disabled={isLoading}
                      className="bg-blue-600 hover:bg-blue-700 px-6"
                    >
                      <Send className="w-4 h-4" />
                    </Button>
                  </form>
                </Form>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConversationPage;
