"use client";
import React, { useState } from "react";
import axios from "axios";
import * as z from "zod";
import { Heading } from "@/components/heading";
import { zodResolver } from "@hookform/resolvers/zod";
import { Form, FormControl, FormField, FormItem } from "@/components/ui/form";
import { useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Loader } from "@/components/loader";

interface Message {
  role: "user" | "bot";
  content: string;
}

const ConversationPage = () => {
  const [indexResponse, setIndexResponse] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

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

  const onIndexSubmit = async (data: any) => {
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
        setIndexResponse(`Repository indexed successfully! Processed ${json.documents_processed} documents.`);
      } else {
        setIndexResponse("Failed to index repository: " + (json.detail || "Unknown error"));
      }
    } catch (error) {
      console.error("Error indexing repository: ", error);
      setIndexResponse("Failed to index repository.");
    }
  };

  const onChatSubmit = async (data: any) => {
    setIsLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/api/chat", { message: data.message });
      setMessages([
        ...messages,
        { role: "user", content: data.message },
        { role: "bot", content: response.data.answer },
      ]);
    } catch (error) {
      console.error("Error in chat: ", error);
    }
    setIsLoading(false);
    chatForm.reset();
  };

  return (
    <div>
      <Heading
        title="RAG Chat with GitHub Repositories"
        description="Index a GitHub repository and chat about its contents using AI."
      />

      {/* GitHub Repository Index Form */}
      <div className="px-4 lg:px-8 mt-4">
        <Form {...indexForm}>
          <form
            onSubmit={indexForm.handleSubmit(onIndexSubmit)}
            className="rounded-lg border w-full p-4 px-3 md:px-6 focus-within:shadow-sm grid grid-cols-12 gap-2"
          >
            <FormField
              name="github_url"
              render={({ field }) => (
                <FormItem className="col-span-12 lg:col-span-10">
                  <FormControl className="m-0 p-0">
                    <Input {...field} placeholder="Enter GitHub repository URL (e.g., https://github.com/user/repo)" />
                  </FormControl>
                </FormItem>
              )}
            />
            <Button className="col-span-12 lg:col-span-2 w-full" type="submit">
              Index Repository
            </Button>
          </form>
        </Form>
        {indexResponse && <p className="mt-4 text-sm text-gray-600">{indexResponse}</p>}
      </div>

      {/* Chat Interaction Form */}
      <div className="px-4 lg:px-8 mt-4">
        <Form {...chatForm}>
          <form
            onSubmit={chatForm.handleSubmit(onChatSubmit)}
            className="rounded-lg border w-full p-4 px-3 md:px-6 focus-within:shadow-sm grid grid-cols-12 gap-2"
          >
            <FormField
              name="message"
              render={({ field }) => (
                <FormItem className="col-span-12 lg:col-span-10">
                  <FormControl className="m-0 p-0">
                    <Input {...field} placeholder="Type your message" />
                  </FormControl>
                </FormItem>
              )}
            />
            <Button
              className="col-span-12 lg:col-span-2 w-full bg-green text-white"
              type="submit"
            >
              Send Message
            </Button>
          </form>
        </Form>
        <div className="space-y-4 mt-4">
          {isLoading && (
            <div className="p-8 rounded-lg w-full flex items-center justify-center bg-muted">
              <Loader />
            </div>
          )}
          {
            messages.length === 0 &&
              indexResponse === "" &&
              !isLoading &&
              null /* This will not render anything */
          }
          <div className="flex flex-col-reverse gap-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`p-8 w-full flex items-start gap-x-8 rounded-lg ${
                  message.role === "user"
                    ? "bg-white border border-black/10"
                    : "bg-muted"
                }`}
              >
                {message.role === "user" ? "User" : "Bot"}
                <p className="text-sm">{message.content}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConversationPage;
